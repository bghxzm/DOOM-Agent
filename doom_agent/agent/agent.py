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
    def __init__(self, debug, train):
        self.debug = debug
        self.train = train


    def run(self):
        if self.debug:
            print(f"Debug Agent!")
        if self.train:
            print(f"Training Agent!")
        if not self.debug and not self.train:
            print(f"Default Agent!")
