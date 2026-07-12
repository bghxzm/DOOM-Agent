"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Coordinate and run the agent.
agent.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import torch
import vizdoom as vzd

from data.relabeler import KILL_REWARD_THRESHOLD
from environment.game import Game
from memory.buffer import Buffer
from model.policy_head import Policy_Head
from model.temporal_transformer import Temporal_Transformer

from stable_baselines3 import PPO


class Agent():
    """
    The integration point: environment + frozen CLIP encoder + sliding
    window buffer + temporal transformer + policy head, running under
    TRAINED weights.  Evaluates sub-goal completion rate, episode
    efficiency and exports the checkpoint bundle for Unity Comparison.
    """
    def __init__(self, config=None, encoder=None):
        self.config = config
        self.encoder = encoder
        self.device = self.config['device']
        self.artifacts_path = Path(self.config['artifacts_path'])
        self.checkpoints_path = self.artifacts_path / "checkpoints"
        self.g = None
        self.buffer = None
        self.transformer = None
        self.policy_head = None
        self.policy_name = None

    def init(self):
        '''
        Game and models are built here rather than __init__ so that the
        export path can use an Agent without paying any startup cost.
        '''
        self.g = Game(path=self.config['scenario_path'])
        self.g.init(resolution=vzd.ScreenResolution.RES_320X240, visible=False)
        self.buffer = Buffer()
        self.buffer.init()
        self.transformer = Temporal_Transformer().to(self.device)
        self.policy_head = Policy_Head(
            hidden_dim=256,
            num_actions=len(self.g.actions)).to(self.device)

    def load_checkpoint(self, policy="ppo"):
        '''
        Both checkpoint formwats end up in the same two modules, so the
        evaluation loop has exactly one path.

        "bc":  bc_policy.pt stores our state dicts directly.
        "ppo": ppo_policy.zip is a Stable Baselines3 archive; the weights
               are pulled back out of the SB3 policy.  The exact inverse
               of the warm-start transfer in ppo_trainer.load_bc_weights.
        '''
        self.policy_name = policy

        if policy == "bc":
            ckpt = torch.load(self.checkpoints_path / "bc_policy.pt",
                             map_location=self.device)
            self.transformer.load_state_dict(ckpt["transformer"])
            self.policy_head.load_state_dict(ckpt["policy_head"])
            print(f"Loaded BC checkpoint "
                  f"(epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.3f})")
        elif policy == "ppo":
            model = PPO.load(self.checkpoints_path / "ppo_policy",
                             device="cpu")
            self.transformer.load_state_dict(
                model.policy.features_extractor.transformer.state_dict())
            with torch.no_grad():
                self.policy_head.linear.weight.copy_(
                    model.policy.action_net.weight)
                self.policy_head.linear.bias.copy_(
                    model.policy.action_net.bias)
            self.transformer.to(self.device)
            self.policy_head.to(self.device)
            print("Loaded PPO checkpoint")
        else:
            raise ValueError(f"Unknown policy '{policy}' -- use bc or ppo")

    def evaluate(self, episodes=20, instruction="kill the monster",
                 frame_repeat=4):
        '''
        Run the trained agent greedily and record the two metrics that define
        the ViZDoom baseline column of the comparison table:

        - completion rate: fraction of episodes where the sub-goal was achieved
          (kills detected the same way as the relabeler: a decision whos raw
          reward exceeds KILL_REWARD_THRESHOLD).
        - episode efficiency: mean decisions-to-completion over SUCCESSFUL
          episodes only.  Averaging in timeouts would poison the number (i.e.,
          A policy that fails fast would look "efficient").

        Greedy selection (argmax) because this is evaluation: we want to see
        what the policy actually learned and not exploration noise.
        eval() turns dropout off; no_grad() skips autograd.
        '''
        self.transformer.eval()
        self.policy_head.eval()
        goal_emb = self.encoder.encode_subgoal(instruction)

        rows = []
        for i in range(episodes):
            self.g.game.new_episode()
            self.buffer.reset()
            decisions, success, to_kill = 0, False, None

            while not self.g.game.is_episode_finished():
                state = self.g.game.get_state()
                frame_emb = self.encoder.encode_frame(state.screen_buffer)
                health = self.g.game.get_game_variable(
                    vzd.GameVariable.HEALTH)
                ammo = self.g.game.get_game_variable(
                    vzd.GameVariable.AMMO2)
                armor = self.g.game.get_game_variable(
                    vzd.GameVariable.ARMOR)

                self.buffer.push(frame_emb, goal_emb, health, ammo, armor)
                window = self.buffer.get_window().to(self.device)

                with torch.no_grad():
                    logits = self.policy_head.forward(
                        self.transformer.forward(window))
                    action_idx = self.policy_head.select_action(
                        logits, greedy=True)

                    reward = self.g.game.make_action(
                        self.g.actions[action_idx], frame_repeat)
                    decisions += 1
                    if reward > KILL_REWARD_THRESHOLD:
                        success, to_kill = True, decisions

            total = self.g.game.get_total_reward()
            rows.append({
                "episode": i + 1,
                "instruction": instruction,
                "success": success,
                "decisions": decisions,
                "decisions_to_kill": to_kill,
                "total_reward": total,
            })
            print(f"Episode {i+1:3d} | "
                  f"{'KILL' if success else 'timeout':>7} | "
                  f"decisions {decisions:3d} | reward {total:7.1f}")

        self.report(rows, instruction, episodes)

    def report(self, rows, instruction, episodes):
        df = pd.DataFrame(rows)
        completion_rate = df["success"].mean()
        successes = df[df["success"]]
        efficiency = (successes["decisions_to_kill"].mean()
                      if len(successes) else float("nan"))

        print(f"\n=== {self.policy_name.upper()} | '{instruction}' | "
              f"{episodes} episodes ===")
        print(f"Sub-goal completion rate: {completion_rate:.2f}")
        print(f"Episode efficiency:       {efficiency:.1f} decisions "
              f"(successes only, n={len(successes)})")
        print(f"Mean total reward:        {df['total_reward'].mean():.1f}")

        out = self.artifacts_path/ f"eval_{self.policy_name}_{self.g.game_cfg}.xlsx"
        df.to_excel(out, index=False, sheet_name="Evaluation")
        print(f"Per-episode results saved to {out}")

    def export_for_unity(self):
        '''
        Bundle everything needed to run this policy in Unity without retraining.
        This manifest is the contract: if any value in it differs between two
        environments the "no retraining" comparison will measure pipeline
        mismatch instead of generalization.
        '''
        export_path = self.artifacts_path / "export"
        export_path.mkdir(parents=True, exist_ok=True)

        bc_ckpt = self.checkpoints_path / "bc_policy.pt"
        ppo_ckpt = self.checkpoints_path / "ppo_policy.zip"
        for ckpt in (bc_ckpt, ppo_ckpt):
            if ckpt.exists():
                shutil.copy2(ckpt, export_path / ckpt.name)

        instructions = []
        if bc_ckpt.exists():
            instructions = torch.load(
                bc_ckpt, map_location="cpu")["instructions"]

        manifest = {
            "encoder": {
                "clip_model": self.config['model'],
                "pretrained": self.config['pretrained'],
                "frame_embedding_dim": 512,
                "goal_embedding_dim": 512,
                "12_normalized": True,
            },
            "observation": {
                "timestep_dim": 1027,
                "concat_order": ["frame_emb", "goal_emb", "hud_vec"],
                "hud_fields": ["health", "ammo", "armor"],
                "hud_normalization": "divide by 200, clip-free",
                "window_size": 8,
                "padding": "zero vectors at window start",
            },
            "model": {
                "hidden_dim": 256, "nhead": 4, "num_layers": 2,
                "dim_feedforward": 512,
                "position_encoding": "sinusoidal",
                "masking": "casual",
                "output": "final position hidden state",
            },
            "actions": {
                "buttons": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
                "table": "all 2^n button combinations, "
                         "itertools.product(False, True) order",
                "frame_repeat": 4,
            },
            "instructions": instructions,
            "checkpoints": {
                "bc": bc_ckpt.name if bc_ckpt.exists() else None,
                "ppo": ppo_ckpt.name if ppo_ckpt.exists() else None,
            },
            "metrics": {
                "completion_rate":
                    "succesful episodes / total episodes",
                "episode_efficiency":
                    "mean decisions to sub-goal completion, "
                    "succesful episodes only",
            },
        }
        with open(export_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Export bundle written to {export_path}")
