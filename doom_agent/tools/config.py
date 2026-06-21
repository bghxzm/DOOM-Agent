"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Agent configuration states.
config.py
"""

import sys
import argparse


class Config():
    """
    Configuration states.
    """
    def __init__(self):
        self.agent_args = set('atd')
        self.agent = ""
        self.debug_agent = False
        self.train_agent = False


    def print_help(self, option):
        help_msg=(
            "Usage:\n"
            "  --agent=<options>\n"
            "\n"
            "  a    default\n"
            "  d    debug\n"
            "  t    training\n"
        )

        if option is None:
            print("No arguments provided!")
        else:
            print(f"\nInvalid option: {option}\n")

        print(help_msg)
        sys.exit()


    def args(self):
        """
        Parse through the arguments input at runtime.
        """
        parser = argparse.ArgumentParser("")
        parser.add_argument("--agent", type=str, nargs='?', const="a",
                            default="")
        args = parser.parse_args()

        self.agent = args.agent
        if not self.agent:
            self.print_help(None)
        elif set(self.agent).issubset(self.agent_args):
            self.debug_agent = 'd' in self.agent
            self.train_agent = 't' in self.agent
        else:
            self.print_help(args.agent)
