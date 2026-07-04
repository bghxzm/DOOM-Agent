"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

ViZDoom wrapper + HUD collection
vizdoom_env.py
"""

from environment.episode import Episode
from environment.game import Game
from model.policy_head import Policy_Head
# from random import choice
from time import sleep
import vizdoom as vzd


class ViZDoom_Env():
    """
    ViZDoom wrapper + HUD collection
    """
    def __init__(self, config=None, encoder=None, buffer=None,
                 transformer=None):
        self.config = config
        self.encoder = encoder
        self.buffer = buffer
        self.transformer = transformer
        self.scenario_path = self.config['scenario_path']
        self.artifacts_path = self.config['artifacts_path']
        self.eps = Episode()
        self.g = Game(path=self.scenario_path)
        self.policy_head = None

    def init(self):
        print("ViZDoom Env!")
        # self.eps = Episode()
        # self.g = Game(path=self.scenario_path)
        self.g.init()
        self.policy_head = Policy_Head(hidden_dim=256,
                                       num_actions=len(self.g.actions))
        self.policy_head.to(self.config['device'])

    def run_default_scenario(self, episodes=1, sleep_time=0.028):
        """
        Run the default scenario game loop.
        """
        for i in range(episodes):
            print(f"Episode #{i+1}")

            self.g.game.new_episode()
            self.buffer.reset()

            while not self.g.game.is_episode_finished():
                state = self.g.game.get_state()
                assert state is not None

                frame = state.screen_buffer # shape (3, H, W), uint8
                frame_emb = self.encoder.encode_frame(frame)
                goal_emb = self.encoder.encode_subgoal("find the red door")

                if self.config['debug']:
                    print(frame_emb.shape)  # should be torch.Size([512]) ✔
                    print(goal_emb.shape)   # should be torch.Size([512]) ✔
                    print(frame_emb.norm()) # should be approx. 1.0 ✔ (1.0000)
                    print(goal_emb.norm())  # should be approx. 1.0 ✔ (1.)

                health = self.g.game.get_game_variable(vzd.GameVariable.HEALTH)
                ammo = self.g.game.get_game_variable(vzd.GameVariable.AMMO2)
                armor = self.g.game.get_game_variable(vzd.GameVariable.ARMOR)

                self.buffer.push(frame_emb, goal_emb, health, ammo, armor)
                window = self.buffer.get_window().to(self.config['device'])

                if self.config['debug']:
                    print(window.shape) # torch.Size([8, 1027]) ✔
                    print(window.dtype) # torch.float32 ✔

                    # Confirms that padding is working.  The first 7 rows
                    # should be all zeros (tensor(0.)) and only the last
                    # row should have data.
                    print("window[0].sum",window[0].sum()) # ✔

                out = self.transformer.forward(window)

                if self.config['debug']:
                    print(out.shape) # torch.Size([256]) ✔
                    print(out.dtype) # torch.float32 ✔

                # logits shape should be 8 because we are generating every
                # possible combination of button presses.  For the basic
                # scenario (3 buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK) that is
                # 2^3 = 8 combinations. The policy head picks the combination
                # with the highest score:
                # [F, F, F] # do nothing
                # [F, F, T] # attack
                # [F, T, F] # move right
                # [F, T, T] # move right + attack
                # [T, F, F] # move left
                # [T, F, T] # move left + attack
                # [T, T, F] # move left + right (nothing happens)
                # [T, T, T] # move left + right + attack
                logits = self.policy_head.forward(out)
                print(logits.shape) # torch.Size([8]) ✔
                print(logits.dtype) # torch.float32 ✔

                action_idx = self.policy_head.select_action(logits)
                print(type(action_idx)) # <class 'int'> ✔
                print(0 <= action_idx < len(self.g.actions)) # True ✔

                action = self.g.actions[action_idx]

                # Makes a random action and save the reward.
                # reward = self.g.game.make_action(choice(self.g.actions))
                reward = self.g.game.make_action(action)

                self.eps.log_episode(i+1, self.g.game, state, reward)

                # Sleep some time because processing is too fast to watch.
                if sleep_time > 0:
                    sleep(sleep_time)

        self.eps.output_episode(game_cfg=self.g.game_cfg, path=self.artifacts_path)
        self.eps.clean_episode()
        self.g.game.close()

    def _test_basic_loop(self, mode, episodes=3, steps=10, frame_skip=1):
        """
        https://github.com/Farama-Foundation/ViZDoom/blob/main/tests/test_basic_loop.py
        """
        game = vzd.DoomGame()
        game.set_mode(mode)
        game.set_window_visible(False)
        game.set_available_buttons(
            [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
        )
        game.set_episode_start_time(35)
        game.init()

        # Just run a few steps to see if anything crashes
        for _ in range(episodes):
            game.new_episode()
            for _ in range(steps):
                if game.is_episode_finished():
                    break

                if mode in {vzd.Mode.ASYNC_SPECTATOR, vzd.Mode.SPECTATOR}:
                    game.advance_action(frame_skip)
                else:
                    game.make_action([0] * game.get_available_buttons_size(),
                                     frame_skip)

        game.close()

    def test_basic_loop(self):
        """
        https://github.com/Farama-Foundation/ViZDoom/blob/main/tests/test_basic_loop.py
        """
        modes = [
            vzd.Mode.PLAYER,
            vzd.Mode.ASYNC_PLAYER,
            vzd.Mode.SPECTATOR,
            vzd.Mode.ASYNC_SPECTATOR
        ]

        frame_skips = [1, 4]

        for mode in modes:
            for frame_skip in frame_skips:
                print(f"Testing mode: {mode}, frame_skip: {frame_skip}")
                self._test_basic_loop(mode, frame_skip=frame_skip)
