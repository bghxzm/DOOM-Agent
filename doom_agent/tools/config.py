"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Agent configuration states.
config.py
"""

import argparse
import sys
import torch


#
# Model architecture and pre-trained weights.
#
MODELS = [
    ("ViT-B-32", "laion2b_s34b_b79k"), # Smaller ViT - baseline
    ("ViT-L-14", "laion2b_s32b_b82k"), # Larger ViT   - larger size impact
    ("convnext_base_w", "laion2b_s13b_b82k_augreg"), # ConvNet architecture
]


class Config():
    """
    Configuration states.
    """
    def __init__(self):
        self.agent_args = set('atd')
        self.agent = ""
        self.encoder = ""
        self.environment = ""
        self.device = None
        self.model = None

    def print_help(self, option):
        """
        Invalid config help
        """
        help_msg=(
            "Usage:\n"
            "  --agent=<options>\n"
            "\n"
            "  a    default\n"
            "  d    debug\n"
            "  t    training\n"
            "  --encoder\n"
            "  --environment\n"
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
        parser.add_argument("--encoder", type=str, nargs='?', const="a",
                            default="")
        parser.add_argument("--environment", type=str, nargs='?', const="a",
                            default="")
        args = parser.parse_args()

        self.agent = args.agent
        self.encoder = args.encoder
        self.environment = args.environment
        if not self.agent and not self.encoder and not self.environment:
            self.print_help(None)
        elif set(self.agent).issubset(self.agent_args):
            if 'd' in self.agent:
                self.agent = "debug"
            elif 't' in self.agent:
                self.agent = "train"
        elif self.encoder:
            self.encoder = "encoder"
        elif self.environment:
            self.environment = "environment"
        else:
            self.print_help(args)

    def dev(self):
        """
        Configure which torch device to use.
        """
        self.device = (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        print(f"Using device: {self.device}")

    def init(self):
        """
        Initial configuration.
        """
        self.args()
        self.dev()
