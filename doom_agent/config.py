"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Agent configuration states.
config.py
"""

from pathlib import Path
import argparse
import sys
import torch


#
# Model architecture and pre-trained weights.
#
MODELS = {
    "ViT-B-32": "laion2b_s34b_b79k", # Smaller ViT - baseline
    "ViT-L-14": "laion2b_s32b_b82k", # Larger ViT   - larger size impact
    "convnext_base_w": "laion2b_s13b_b82k_augreg", # ConvNet architecture
}

DEVICES = [
    "mps",
    "cuda",
    "cpu"
]


class Config():
    """
    Configuration states.
    """
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config = {
            "model": None,
            "device": None,
            "pretrained": None,
            "artifacts_path": None,
            "cache_path": None,
            "scenario_path": None,
            "encoder": False,
            "environment": False,
            "collect": False,
            "relabel": False,
            "bc": False,
            "ppo": False,
            "eval": False,
            "export": False,
            "policy": "ppo",
            "episodes": 20,
            "epochs": 30,
            "timesteps": 20000,
            "test": False,
            "debug": False
        }

    def init(self):
        """
        Initial configuration.
        """
        self.args()
        if not self.config['device']:
            self.choose_device()
        if not self.config['model']:
            self.choose_model()
        if not self.config['artifacts_path']:
            self.choose_artifacts_path()
        if not self.config['cache_path']:
            self.choose_cache_path()
        if not self.config['scenario_path']:
            self.choose_scenario_path()

    def print_config(self):
        """
        Print config settings.
        """
        print(f"\nCONFIG:")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print(f"\n")

    def print_help(self, option):
        """
        Invalid config help
        """
        help_msg=(
            "Usage:\n"
            "  --encoder\n"
            "  --model=<options>\n"
            "\n"
            "  ViT-B-32\n"
            "  ViT-L-14\n"
            "  convnext_base_w\n"
            "  --device=<options>\n"
            "\n"
            "  mps\n"
            "  cuda\n"
            "  cpu\n"
            "  --environment\n"
            "  --test\n"
            "  --artifacts=<dir>\n"
            "\n"
            "  dir\n"
            "  --cache=<dir>\n"
            "\n"
            "  dir\n"
            "  --scenario=<dir>\n"
            "\n"
            "  dir\n"
            "  --collect\n"
            "  --relabel\n"
            "  --bc\n"
            "  --ppo\n"
            "  --eval\n"
            "  --export\n"
            "  --policy=<bc|ppo>\n"
            "  --episodes=<num>\n"
            "  --epochs=<num>\n"
            "  --timesteps=<num>\n"
            "  --debug\n"
        )
        if option is None:
            print("No arguments provided!")
        else:
            print(f"\nInvalid option: {option}\n")

        print(help_msg)
        sys.exit()

    def choose_model(self, model_name="ViT-B-32"):
        """
        Configure which model to use.
        """
        if model_name not in MODELS:
            self.config['model'] = "ViT-B-32"
            self.config['pretrained'] = MODELS[self.config['model']]
        else:
            self.config['model'] = model_name
            self.config['pretrained'] = MODELS[model_name]

    def choose_device(self, device_name=""):
        """
        Configure which torch device to use.
        """
        if device_name not in DEVICES:
            self.config['device'] = (
                "mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )
        else:
            self.config['device'] = device_name

    def choose_artifacts_path(self, artifacts_path="artifacts"):
        """
        Configure which artifacts path to use.
        """
        path = self.project_root / artifacts_path
        if path.is_dir():
            self.config['artifacts_path'] = path
        else:
            self.config['artifacts_path'] = self.project_root / "artifacts"

    def choose_cache_path(self, cache_path="cache"):
        """
        Configure which cache path to use.
        """
        path = self.project_root / cache_path
        if path.is_dir():
            self.config['cache_path'] = path
        else:
            self.config['cache_path'] = self.project_root / "cache"

    def choose_scenario_path(self, scenario_path="scenarios"):
        """
        Configure which scenario path to use.
        """
        path = self.project_root / scenario_path
        if path.is_dir():
            self.config['scenario_path'] = path
        else:
            self.config['scenario_path'] = self.project_root / "scenarios"

    def args(self):
        """
        Parse through the arguments input at runtime.
        """
        valid = ["encoder", "model", "device", "environment", "test",
                 "artifacts", "cache", "scenario", "collect", "relabel",
                 "bc", "ppo", "eval", "export", "policy", "episodes",
                 "epochs", "timesteps", "debug"]

        parser = argparse.ArgumentParser("")
        for v in valid:
            parser.add_argument("--"+v, type=str, nargs='?', default="")
        args = parser.parse_args()

        for arg in vars(args):
            if arg == "encoder":
                if (getattr(args, arg) == None):
                    self.config['encoder'] = True
            elif arg == "model":
                self.choose_model(getattr(args, arg))
            elif arg == "device":
                self.choose_device(getattr(args, arg))
            elif arg == "environment":
                if (getattr(args, arg) == None):
                    self.config['environment'] = True
            elif arg == "test":
                if (getattr(args, arg) == None):
                    self.config['test'] = True
            elif arg == "artifacts":
                if (getattr(args, arg) != ''):
                    self.choose_artifacts_path(getattr(args, arg))
            elif arg == "cache":
                if (getattr(args, arg) != ''):
                    self.choose_cache_path(getattr(args, arg))
            elif arg == "scenario":
                if (getattr(args, arg) != ''):
                    self.choose_scenario_path(getattr(args, arg))
            elif arg == "collect":
                if (getattr(args, arg) == None):
                    self.config['collect'] = True
            elif arg == "relabel":
                if (getattr(args, arg) == None):
                    self.config['relabel'] = True
            elif arg == "bc":
                if (getattr(args, arg) == None):
                    self.config['bc'] = True
            elif arg == "ppo":
                if (getattr(args, arg) == None):
                    self.config['ppo'] = True
            elif arg == "eval":
                if (getattr(args, arg) == None):
                    self.config['eval'] = True
            elif arg == "export":
                if (getattr(args, arg) == None):
                    self.config['export'] = True
            elif arg == "policy":
                if (getattr(args, arg) != ''):
                    self.config['policy'] = getattr(args, arg)
            elif arg == "episodes":
                if (getattr(args, arg) != ''):
                    self.config['episodes'] = int(getattr(args, arg))
            elif arg == "epochs":
                if (getattr(args, arg) != ''):
                    self.config['epochs'] = int(getattr(args, arg))
            elif arg == "timesteps":
                if (getattr(args, arg) != ''):
                    self.config['timesteps'] = int(getattr(args, arg))
            elif arg == "debug":
                if (getattr(args, arg) == None):
                    self.config['debug'] = True
            else:
                self.print_help(args)
