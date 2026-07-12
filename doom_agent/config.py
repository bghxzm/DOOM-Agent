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

# Button positions in actions_table entries.  Order matches available_buttons
# in scenarios/basic.cfg: [MOVE_LEFT, MOVE_RIGHT, ATTACK].
MOVE_LEFT, MOVE_RIGHT, ATTACK = 0, 1, 2

# basic.wad pays +101 for the kill, -1/tic living, -5 per missed shot.
# With frame_repeat=4 a kill step nets ~+97 and the worst possible non-kill
# step nets ~-9 so anything above this threshold is a kill.
KILL_REWARD_THRESHOLD = 50.0

# Movement runs shorter than this many decisions (4 tics each) are jitter
# from the sticky random policy, not purposeful movement.
MIN_RUN_LENGTH = 3


class Default_Config():
    """
    Configure default paths and states.
    """
    def __init__(self, config=None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.cfg = config
        self.model = self.cfg['model']
        self.pretrained = self.cfg['pretrained']
        self.device = self.cfg['device']
        self.paths = self.cfg['paths']

    def choose_model(self, model_name="ViT-B-32"):
        '''
        Configure which model to use.
        '''
        if model_name not in MODELS:
            self.model = "ViT-B-32"
            self.pretrained = MODELS[self.model]
        else:
            self.model = model_name
            self.pretrained = MODELS[model_name]

    def choose_device(self, device_name=""):
        '''
        Configure which torch device to use.
        '''
        if device_name not in DEVICES:
            self.device = (
                "mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )
        else:
            self.device = device_name

    def configure_path(self, path_idx, path_to_assign):
        '''
        Check if the path exists.  If it does not, then create one.
        '''
        if path_to_assign.is_dir():
            self.paths[path_idx] = path_to_assign
        else:
            self.paths[path_idx].mkdir(parents=True, exist_ok=True)

    def choose_artifacts_path(self, artifacts_path="artifacts"):
        '''
        Configure which artifacts path to use.
        '''
        path = self.project_root / artifacts_path
        self.configure_path('artifacts_path', path)

    def choose_scenarios_path(self, scenarios_path="scenarios"):
        '''
        Configure which scenarios path to use.
        '''
        path = self.project_root / scenarios_path
        self.configure_path('scenarios_path', path)

    def choose_trajectories_path(self, trajectories_path="trajectories"):
        '''
        Configure which trajectories path to use.
        '''
        path = self.paths['artifacts_path'] / trajectories_path
        self.configure_path('trajectories_path', path)

    def choose_relabeled_path(self, relabeled_path="relabeled"):
        '''
        Configure which relabeled path to use.
        '''
        path = self.paths['artifacts_path'] / relabeled_path
        self.configure_path('relabeled_path', path)

    def choose_embeddings_path(self, embeddings_path="embeddings"):
        '''
        Configure which embeddings path to use.
        '''
        path = self.paths['artifacts_path'] / embeddings_path
        self.configure_path('embeddings_path', path)

    def choose_checkpoints_path(self, checkpoints_path="checkpoints"):
        '''
        Configure which checkpoints path to use.
        '''
        path = self.paths['artifacts_path'] / checkpoints_path
        self.configure_path('checkpoints_path', path)

    def choose_ppo_logs_path(self, ppo_logs_path="ppo_logs"):
        '''
        Configure which ppo_logs path to use.
        '''
        path = self.paths['artifacts_path'] / ppo_logs_path
        self.configure_path('ppo_logs_path', path)

    def choose_export_path(self, export_path="export"):
        '''
        Configure which export path to use.
        '''
        path = self.paths['artifacts_path'] / export_path
        self.configure_path('export_path', path)

    def choose_bc_checkpoint_path(self, bc_checkpoint_path="bc_policy"):
        '''
        Configure which behavior cloning checkpoint path to use.
        '''
        bc_policy_pt =  bc_checkpoint_path + ".pt"
        self.paths['bc_checkpoint_path'] = self.paths['checkpoints_path'] / bc_policy_pt

    def choose_ppo_checkpoint_path(self, ppo_checkpoint_path="ppo_policy"):
        '''
        Configure which ppo_policy path to use.
        '''
        ppo_checkpoint_zip =  ppo_checkpoint_path + ".zip"
        self.paths['ppo_checkpoint_path'] = self.paths['checkpoints_path'] / ppo_checkpoint_path
        self.paths['ppo_checkpoint_zip'] = self.paths['ppo_checkpoint_path'] / ppo_checkpoint_zip


class Config_Args():
    """
    Config args.
    """
    def __init__(self, config=None, default_config=None):
        self.cfg = config
        self.dc = default_config

    def print_help(self, option):
        '''
        Invalid config help
        '''
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
            "\n"
            "  dir\n"
            "  --artifacts_path=<dir>\n"
            "  --scenarios_path=<dir>\n"
            "  --trajectories_path=<dir>\n"
            "  --relabeled_path=<dir>\n"
            "  --embeddings_path=<dir>\n"
            "  --checkpoints_path=<dir>\n"
            "  --ppo_logs_path=<dir>\n"
            "  --export_path=<dir>\n"
            "  --bc_checkpoint_path=<dir>\n"
            "  --ppo_checkpoint_path=<dir>\n"
            "\n"
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
            "  --test\n"
        )

        if option is None:
            print("No arguments provided!")
        else:
            print(f"\nInvalid option: {option}\n")

        print(help_msg)
        sys.exit()

    def parse_args(self):
        '''
        Parse through the arguments input at runtime.
        '''
        valid = ["encoder", "model", "device", "environment",
                 "artifacts_path", "scenarios_path", "trajectories_path",
                 "relabeled_path", "embeddings_path", "checkpoints_path",
                 "ppo_logs_path", "export_path", "bc_checkpoint_path",
                 "ppo_checkpoint_path", "collect", "relabel", "bc", "ppo",
                 "eval", "export", "policy", "episodes", "epochs", "timesteps",
                 "debug", "test"]

        parser = argparse.ArgumentParser("")
        for v in valid:
            parser.add_argument("--"+v, type=str, nargs='?', default="")
        args = parser.parse_args()

        for arg in vars(args):
            if arg == "encoder":
                if (getattr(args, arg) == None):
                    self.cfg['encoder'] = True
            elif arg == "model":
                self.dc.choose_model(getattr(args, arg))
            elif arg == "device":
                self.dc.choose_device(getattr(args, arg))
            elif arg == "environment":
                if (getattr(args, arg) == None):
                    self.cfg['environment'] = True
            elif arg == "artifacts_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_artifacts_path(getattr(args, arg))
            elif arg == "scenarios_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_scenarios_path(getattr(args, arg))
            elif arg == "trajectories_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_trajectories_path(getattr(args, arg))
            elif arg == "relabeled_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_relabeled_path(getattr(args, arg))
            elif arg == "embeddings_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_embeddings_path(getattr(args, arg))
            elif arg == "checkpoints_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_checkpoints_path(getattr(args, arg))
            elif arg == "ppo_logs_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_ppo_logs_path(getattr(args, arg))
            elif arg == "export_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_export_path(getattr(args, arg))
            elif arg == "bc_checkpoint_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_bc_checkpoint_path(getattr(args, arg))
            elif arg == "ppo_checkpoint_path":
                if (getattr(args, arg) != ''):
                    self.dc.choose_ppo_checkpoint_path(getattr(args, arg))
            elif arg == "collect":
                if (getattr(args, arg) == None):
                    self.cfg['collect'] = True
            elif arg == "relabel":
                if (getattr(args, arg) == None):
                    self.cfg['relabel'] = True
            elif arg == "bc":
                if (getattr(args, arg) == None):
                    self.cfg['bc'] = True
            elif arg == "ppo":
                if (getattr(args, arg) == None):
                    self.cfg['ppo'] = True
            elif arg == "eval":
                if (getattr(args, arg) == None):
                    self.cfg['eval'] = True
            elif arg == "export":
                if (getattr(args, arg) == None):
                    self.cfg['export'] = True
            elif arg == "policy":
                if (getattr(args, arg) != ''):
                    self.cfg['policy'] = getattr(args, arg)
            elif arg == "episodes":
                if (getattr(args, arg) != ''):
                    self.cfg['episodes'] = int(getattr(args, arg))
            elif arg == "epochs":
                if (getattr(args, arg) != ''):
                    self.cfg['epochs'] = int(getattr(args, arg))
            elif arg == "timesteps":
                if (getattr(args, arg) != ''):
                    self.cfg['timesteps'] = int(getattr(args, arg))
            elif arg == "debug":
                if (getattr(args, arg) == None):
                    self.cfg['debug'] = True
            elif arg == "test":
                if (getattr(args, arg) == None):
                    self.cfg['test'] = True
            else:
                self.print_help(args)


class Config():
    """
    Config values.
    """
    def __init__(self):
        self.config = {
            "model": None,
            "device": None,
            "pretrained": None,
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
            "debug": False,
            "test": False,
            "paths": {
                "artifacts_path": None,
                "scenarios_path": None,
                "trajectories_path": None,
                "relabeled_path": None,
                "embeddings_path": None,
                "checkpoints_path": None,
                "ppo_logs_path": None,
                "export_path": None,
                "bc_checkpoint_path": None,
                "ppo_checkpoint_path": None,
                "ppo_checkpoint_zip": None,
            },
            "game": {
                "MOVE_LEFT": 0,
                "MOVE_RIGHT": 1,
                "ATTACK": 2,
                "KILL_REWARD_THRESHOLD": 50.0,
                "MIN_RUN_LENGTH": 3,
            }
        }
        self.dc = Default_Config(self.config)
        self.ca = Config_Args(self.config, self.dc)

    def init(self):
        '''
        Initial configuration.
        '''
        self.ca.parse_args()
        if not self.config['device']:
            self.dc.choose_device()
        if not self.config['model']:
            self.dc.choose_model()
        if not self.config['paths']['artifacts_path']:
            self.dc.choose_artifacts_path()
        if not self.config['paths']['scenarios_path']:
            self.dc.choose_scenarios_path()
        if not self.config['paths']['trajectories_path']:
            self.dc.choose_trajectories_path()
        if not self.config['paths']['relabeled_path']:
            self.dc.choose_relabeled_path()
        if not self.config['paths']['embeddings_path']:
            self.dc.choose_embeddings_path()
        if not self.config['paths']['checkpoints_path']:
            self.dc.choose_checkpoints_path()
        if not self.config['paths']['ppo_logs_path']:
            self.dc.choose_ppo_logs_path()
        if not self.config['paths']['export_path']:
            self.dc.choose_export_path()
        if not self.config['paths']['bc_checkpoint_path']:
            self.dc.choose_bc_checkpoint_path()
        if not self.config['paths']['ppo_checkpoint_path']:
            self.dc.choose_ppo_checkpoint_path()

    def print_config(self):
        '''
        Print config settings.
        '''
        print(f"\nRunning with CONFIG settings:")
        for key, value in self.config.items():
            if key == "paths" or key == "game":
                for key1, value1 in value.items():
                    print(f"{key1}: {value1}")
            else:
                print(f"{key}: {value}")
        print(f"\n")
