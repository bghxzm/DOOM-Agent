"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Scripted play trajectory collection.
collector.py
"""

from random import randrange, random
import pickle

import vizdoom as vzd
from environment.game import Game


class Collector():
    """
    Scripted play trajectory collection.

    Plays episodes with a scripted (random + sticky) policy and records
    every decision step to disk.  This raw data is the input to hindsight
    instruction relabeling and behavior cloning.
    """
    def __init__(self, config=None):
        self.config = config
        self.artifacts_path = self.config['paths']['artifacts_path']
        self.scenarios_path = self.config['paths']['scenarios_path']
        self.trajectories_path = self.config['paths']['trajectories_path']
        self.g = Game(config=self.config)
        self.last_action_idx = None

    def init(self):
        '''
        Collection runs headless (visible=False) and not throttled to real-time
        rendering.  Resolution is dropped to 320x240 because CLIP preprocessing
        resizes/center-crops to 224x224.
        '''
        self.g.init(resolution=vzd.ScreenResolution.RES_320X240, visible=False)

    def scripted_action(self, sticky=0.7):
        '''
        Rule-based random policy with "sticky" actions.

        A purely random policy re-rolls the action at every decision.
        This makes the agent jitter in place and cover very little of
        the map.  Sticky actions repeat the previous action with
        probability `sticky`.  This produces longer, more purposeful
        movement segments that actually reach doors and items which is
        much better to use for hindsight relabeling.
        '''
        if self.last_action_idx is not None and random() < sticky:
            return self.last_action_idx
        self.last_action_idx = randrange(len(self.g.actions))
        return self.last_action_idx

    def collect(self, episodes=20, frame_repeat=4):
        '''
        Play a number episode(s) and save one .pkl trajectory file for each.

        Each decision step records the tuple from the plan: (frame,
        subgoal, hud, action_taken) + reward + raw game variables.
        These variables are needed by hindsight relabeling in order
        to decide what the agent actually achieved.
        '''
        for i in range(episodes):
            print(f"Collecting episode #{i+1}/{episodes}")
            self.g.game.new_episode()
            self.last_action_idx = None
            steps = []

            while not self.g.game.is_episode_finished():
                state = self.g.game.get_state()
                assert state is not None

                # .copy() so the stored frame owns its memory.  VizDoom
                # may reuse the underlying screen buffer on the next step
                # and without a copy every recorded step could end up pointing
                # at the same (latest) frame.
                frame = state.screen_buffer.copy() # (3, H, W) uint8

                health = self.g.game.get_game_variable(vzd.GameVariable.HEALTH)
                ammo = self.g.game.get_game_variable(vzd.GameVariable.AMMO2)
                armor = self.g.game.get_game_variable(vzd.GameVariable.ARMOR)

                action_idx = self.scripted_action()
                action = self.g.actions[action_idx]
                reward = self.g.game.make_action(action, frame_repeat)

                steps.append({
                    "frame": frame,
                    "subgoal": None, # filled in by the relabeler
                    "hud": (health, ammo, armor),
                    "action": action_idx,
                    "reward": reward,
                    "game_variables": state.game_variables
                })

            trajectory = {
                "scenario": self.g.game_cfg,
                "actions_table": self.g.actions,
                "frame_repeat": frame_repeat,
                "total_reward": self.g.game.get_total_reward(),
                "steps": steps
            }
            self.save_trajectory(trajectory, i)

        self.g.game.close()

    def save_trajectory(self, trajectory, index):
        '''
        One file per episode keeps memory flat during collection and lets
        hindsight instruction relabeling and behavior cloning stream episodes
        from a disk instead of loading everything.
        '''
        out = f"{self.g.game_cfg}_ep{index:04d}.pkl"
        output = self.trajectories_path / out
        with open(output, "wb") as f:
            pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {output} ({len(trajectory['steps'])} steps, "
              f"total_reward={trajectory['total_reward']})")

    def load_trajectory(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def inspect(self):
        '''
        Load every saved trajectory back from a disk and print a summary
        to prove the data round-trips correctly.
        '''
        for path in sorted(self.trajectories_path.glob("*.pkl")):
            t = self.load_trajectory(path)
            first = t["steps"][0]
            print(f"{path.name}: {len(t['steps'])} steps | "
                  f"hud {first['hud']} | "
                  f"action {first['action']} | "
                  f"total_reward {t['total_reward']}")
