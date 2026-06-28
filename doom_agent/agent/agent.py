"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Coordinate and run the agent.
agent.py
"""


class Agent():
    """
    Agent
    """
    def __init__(self):
        self.mode = None

    def init(self, mode):
        """
        Initialize the agent and set it's mode.
        """
        self.mode = mode

        if mode == "debug":
            print(f"Debug Agent!")
        elif mode == "train":
            print(f"Training Agent!")
        else:
            print(f"Default Agent!")
