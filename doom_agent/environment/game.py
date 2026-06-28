"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

ViZDoom game handler
game.py
"""

import itertools
import os
from argparse import ArgumentParser
import vizdoom as vzd


class Game():
    """
    Game Handler
    """
    def __init__(self):
        self.game = None
        self.config = None
        self.actions = []

    def init(self, path="../scenarios/", cfg="basic.cfg",
             resolution=vzd.ScreenResolution.RES_640X480,
             visible=True):
        """
        Initialize the game.
        """
        self.config = cfg
        default_config = os.path.join(path, self.config)

        parser = ArgumentParser("ViZDoom Scenarios")
        parser.add_argument(
            dest="config",
            default=default_config,
            nargs="?",
            help="Path to the configuration file of the scenario."
            " Please see "
            "../scenarios/*cfg for more scenarios."
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
        self.game.load_config(run_config)

        self.game.set_screen_resolution(resolution)
        self.game.set_window_visible(visible)

        # Sets format to 3 channels of 8-bit values in RGB order [Channels,Height,Width]
        self.game.set_screen_format(vzd.ScreenFormat.CRCGCB)
        self.game.init()
