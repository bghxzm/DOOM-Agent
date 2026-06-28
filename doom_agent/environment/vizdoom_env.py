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
from random import choice
from time import sleep
import vizdoom as vzd


class ViZDoom_Env():
    """
    ViZDoom wrapper + HUD collection
    """
    def __init__(self, config=None):
        self.config = config
        self.scenario_path = self.config['scenario_path']
        self.artifacts_path = self.config['artifacts_path']
        self.eps = None
        self.g = None

    def init(self):
        print("ViZDoom Env!")
        self.eps = Episode()
        self.g = Game(path=self.scenario_path)
        self.g.init()

    def run_default_scenario(self, episodes=10, sleep_time=0.028):
        """
        Run the default scenario game loop.
        """
        for i in range(episodes):
            print(f"Episode #{i+1}")

            # Not needed for the first episode but the loop is nicer.
            self.g.game.new_episode()
            while not self.g.game.is_episode_finished():
                # Gets the state and possibly to something with it.
                state = self.g.game.get_state()
                assert state is not None

                # Makes a random action and save the reward.
                reward = self.g.game.make_action(choice(self.g.actions))
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
