"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

ViZDoom wrapper + HUD collection for testing.
vizdoom_env_test.py
"""

from environment.episode import Episode
from environment.game import Game
from model.policy_head import Policy_Head
# from random import choice
from time import sleep
import vizdoom as vzd


class ViZDoom_Env_Test():
    """
    ViZDoom wrapper + HUD collection for testing.
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
        self.g.init()
        self.policy_head = Policy_Head(hidden_dim=256,
                                       num_actions=len(self.g.actions))
        self.policy_head.to(self.config['device'])

    def _test_basic_loop(self, mode, episodes=3, steps=10, frame_skip=1):
        '''
        https://github.com/Farama-Foundation/ViZDoom/blob/main/tests/test_basic_loop.py
        '''
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
        '''
        https://github.com/Farama-Foundation/ViZDoom/blob/main/tests/test_basic_loop.py
        '''
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
