"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

ViZDoom game handler
game.py
"""

from argparse import ArgumentParser
from pathlib import Path
import itertools
import vizdoom as vzd


class Game():
    """
    Game Handler
    """
    def __init__(self, path="./scenarios"):
        self.path = Path(path)
        self.game = None
        self.game_cfg = None
        self.actions = []

    def init(self, game_cfg="basic", resolution=vzd.ScreenResolution.RES_640X480,
             visible=True):
        """
        Initialize the game.
        """
        self.game_cfg = game_cfg
        cfg_file = self.game_cfg + ".cfg"
        default_config = self.path / cfg_file

        parser = ArgumentParser("ViZDoom Scenarios")
        parser.add_argument(
            dest="config",
            default=default_config,
            nargs="?",
            help="Path to the configuration file of the scenario."
            " Please see "
            "./scenarios/*cfg for more scenarios."
        )
        args, _ = parser.parse_known_args()
        run_config = args.config

        self.game = vzd.DoomGame()

        # Creates all possible actions depending on how many buttons there are.
        actions_num = self.game.get_available_buttons_size()
        for perm in itertools.product([False, True], repeat=actions_num):
            self.actions.append(list(perm))

        # Choose scenario config file you wish to watch.
        # Don't load two configs cause the second will overwrite the first one.
        # Multiple config files are ok but combining these ones doesn't make much sense.
        self.game.load_config(str(run_config))

        self.game.set_screen_resolution(resolution)
        self.game.set_window_visible(visible)

        # Sets format to 3 channels of 8-bit values in RGB order [Channels,Height,Width]
        self.game.set_screen_format(vzd.ScreenFormat.CRCGCB)
        self.game.init()
