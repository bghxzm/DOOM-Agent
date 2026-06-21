"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

main.py
"""

from tools.config import Config
from agent.agent import Agent


def main():
    cfg = Config()
    cfg.args()

    if cfg.agent:
        print("Running Agent!")
        a = Agent(cfg.debug_agent, cfg.train_agent)
        a.run()

if __name__ == "__main__":
    main()
