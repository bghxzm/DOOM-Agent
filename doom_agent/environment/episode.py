"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

ViZDoom episode handler
episode.py
"""

import pandas as pd
import vizdoom as vzd


class Episode():
    """
    Episode Handler
    """
    def __init__(self):
        self.episode_log = {
            "num": [],
            "states": [],
            "variables": [],
            "actions": [],
            "rewards": [],
            "total_reward": [],
            "health": [],
            "ammo": [],
            "armor": [],
            "rgb_buffer": [],
        }

    def log_episode(self, num, game, state, reward):
        '''
        Log all relevant information from an episode.
        '''
        performed_action = game.get_last_action()
        total_reward = game.get_total_reward()
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
        armor = game.get_game_variable(vzd.GameVariable.ARMOR)
        rgb_screen = state.screen_buffer

        self.episode_log["num"].append(num)
        self.episode_log["states"].append(state.number)
        self.episode_log["variables"].append(state.game_variables)
        self.episode_log["actions"].append(performed_action)
        self.episode_log["rewards"].append(reward)
        self.episode_log["total_reward"].append(total_reward)
        self.episode_log["health"].append(health)
        self.episode_log["ammo"].append(ammo)
        self.episode_log["armor"].append(armor)
        self.episode_log["rgb_buffer"].append(rgb_screen)

        print(f"Episode #{num}")
        print(f"State #{state.number}")
        print("Game Variables:", state.game_variables)
        print("Performed action:", performed_action)
        print("Last Reward:", reward)
        print("Total Reward:", total_reward)
        print("Health:", health)
        print("Ammo:", ammo)
        print("Armor:", armor)
        print("=====================")
        # print("Array Shape:", rgb_screen.shape) # (Channel,Height,Width)
        # print("Data Type:", rgb_screen.dtype)   # Output will be uint8 (values 0-255)

    def show_episode(self):
        '''
        Show all of the episode information.
        '''
        for (key, value) in self.episode_log.items():
            print(f"{key}: {value}")

    def output_episode(self, config):
        """
        Show all of the scenario information.
        """
        output_log = dict(list(self.episode_log.items())[:-1])
        df = pd.DataFrame(output_log)
        cfg_fmt = config.replace(".", "_")
        df.to_excel(f'artifacts/{cfg_fmt}_episode.xlsx', index=False, sheet_name="Episode")

    def clean_episode(self):
        '''
        Empty the episode of all values.
        '''
        for key, value in self.episode_log.items():
            self.episode_log[key] = []
