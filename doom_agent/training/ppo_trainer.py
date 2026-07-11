"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Proximal Policy Optimization via Stable Baselines3
ppo_trainer.py
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import vizdoom as vzd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.game import Game
from memory.buffer import Buffer
from model.temporal_transformer import Temporal_Transformer


class Doom_Gym_Env(gym.Env):
    """
    Gymnasium-compatible wrapper around the ViZDoom game.

    The wrapper owns the frozen CLIP encoder and the sliding-window buffer.
    From Stable Baselines3's point of view the observation is simply a
    [8, 1027] float box; the same window the transformer sees during
    behavior cloning.  SB3 never touches pixels so the CLIP boundary stays
    intact.
    """
    metadata = {"render_modes": []}

    def __init__(self, config=None, encoder=None,
                 instruction="kill the monster", frame_repeat=4):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.instruction = instruction
        self.frame_repeat = frame_repeat

        self.g = Game(path=self.config['scenario_path'])
        self.g.init(resolution=vzd.ScreenResolution.RES_320X240, visible=False)

        self.buffer = Buffer()
        self.buffer.init()

        # The goal is fixed for a training run, so encode it once.
        self.goal_emb = self.encoder.encode_subgoal(self.instruction)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.buffer.window_size, self.buffer.timestep_dim),
            dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.g.actions))
        self.last_obs = None

    def _observe(self):
        '''
        Encode the current frame and push a timestep into the buffer.
        Return the window and the identical path (Buffer.push / get_window)
        used in BC training and the live agent loop.
        '''
        state = self.g.game.get_state()
        frame_emb = self.encoder.encode_frame(state.screen_buffer)
        health = self.g.game.get_game_variable(vzd.GameVariable.HEALTH)
        ammo = self.g.game.get_game_variable(vzd.GameVariable.AMMO2)
        armor = self.g.game.get_game_variable(vzd.GameVariable.ARMOR)

        self.buffer.push(frame_emb, self.goal_emb, health, ammo, armor)
        self.last_obs = self.buffer.get_window().numpy()
        return self.last_obs

    def _shape_reward(self, raw_reward):
        '''
        basic.wad already implements the reward structure from the plan:
        dense shaping (living -1/tic, missed shot -5) pushes towards
        efficient goal completion, and the sparse +101 kill bonus is the
        sub-goal completion reward.  Dividing by 100 scales returns to
        roughly [-1, +1], where PPO's value loss and advantage estimates
        are numerically well behaved.

        For maze scenarios, this hook is where distance-based shaping from
        POSITION_X/Y game variables should be added.
        '''
        return raw_reward / 100.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.g.game.new_episode()
        self.buffer.reset()
        return self._observe(), {}

    def step(self, action_idx):
        raw_reward = self.g.game.make_action(
            self.g.actions[action_idx], self.frame_repeat)
        reward = self._shape_reward(raw_reward)

        terminated = self.g.game.is_episode_finished()
        # After the final tic ViZDoom has no state to observe, so the
        # last window is returned again.  SB3 ignores the observation
        # paired with terminated=True except for bookkeeping.
        obs = self.last_obs if terminated else self._observe()
        return obs, reward, terminated, False, {}

    def close(self):
        self.g.game.close()


class Transformer_Extractor(BaseFeaturesExtractor):
    """
    Adapter that installs the Temporal_Transformer as SB3's features extractor.
    SB3's ActorCriticPolicy pipeline is:

        obs0 -> features_extractor -> mlp_extractor -> action_net / value_net

    With net_arch=dict(pi=[], vf=[]) the mlp_extractor is empty so action_net
    is exactly Linear(256, num_actions).  Policy_Head.linear is the same shape
    which makes the BC weight transfer a direct layer-for-layer copy.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.transformer = Temporal_Transformer()

    def forward(self, observations):
        # observations: [B, 8, 1027] -> [B, 256]
        return self.transformer.forward(observations)


class PPO_Trainer():
    """
    Proximal Policy Optimization via Stable Baselines3.  Warm-started
    from the behavior-cloning checkpoint.

    BC's limitation is that it can only be as purposeful as its demonstrator.
    For example, if our demonstrator is a random walk, PPO fixes this by
    optimizing the actual reward: the policy is credited for killing the
    monster and not for imitating noise.
    """
    def __init__(self, config=None, encoder=None):
        self.config = config
        self.encoder = encoder
        self.artifacts_path = Path(self.config['artifacts_path'])
        self.checkpoints_path = self.artifacts_path / "checkpoints"
        self.logs_path = self.artifacts_path / "ppo_logs"
        self.model = None

    def init(self):
        print("Proximal Policy Optimization Trainer!")
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

    def load_bc_weights(self, checkpoint_path):
        '''
        Transfer the BC-trained weights into the SB3 policy:

            transformer -> policy.features_extractor.transformer
            policy head -> policy.action_net (Linear 256 -> num_actions)

        The value_net is deliberately NOT loaded BC never saw rewards
        so no value function exists to transfer.  It trains from scratch
        which is why the learning rate is small: early value estimates are
        garbage and a large step size would let them wreck the BC policy
        before calibration.
        '''
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loading BC checkpoint (epoch {ckpt['epoch']}, "
              f"val acc {ckpt['val_acc']:.3f})")

        self.model.policy.features_extractor.transformer.load_state_dict(
            ckpt["transformer"])
        with torch.no_grad():
            self.model.policy.action_net.weight.copy_(
                ckpt["policy_head"]["linear.weight"])
            self.model.policy.action_net.bias.copy_(
                ckpt["policy_head"]["linear.bias"])

    def train(self, total_timesteps=20000, instruction="kill the monster"):
        '''
        Monitor wraps the env and appends one line per episode (reward,
        length, wall time) to a CSV in artifacts/ppo_logs/*
        This file is the "reward improves over training" evidence.
        '''
        env = Monitor(
            Doom_Gym_Env(config=self.config, encoder=self.encoder,
                         instruction=instruction),
            str(self.logs_path / "ppo"))

        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                features_extractor_class=Transformer_Extractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[], vf=[]),
            ),
            learning_rate=3e-5,  # small: protect the BC warm start
            n_steps=256,         # rollout length between updates
            batch_size=64,
            ent_coef=0.01,       # mild exploration pressure
            verbose=1,
            device=self.config['device']
        )

        bc_checkpoint = self.checkpoints_path / "bc_policy.pt"
        if bc_checkpoint.exists():
            self.load_bc_weights(bc_checkpoint)
        else:
            print("Training Proximal Policy Optimization from scratch.")

        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.checkpoints_path / "ppo_policy")
        print(f"Saved PPO checkpoint to "
              f"{self.checkpoints_path / 'ppo_policy.zip'}")
        env.close()
