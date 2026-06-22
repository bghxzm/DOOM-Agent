"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

main.py
"""

from tools.config import Config
from agent.agent import Agent
from data.relabeler import Relabeler
from data.collector import Collector
from encoder.clip_encoder import CLIP_Encoder
from environment.vizdoom_env import ViZDoom_Env
from memory.buffer import Buffer
from model.policy_head import Policy_Head
from model.temporal_transformer import Temporal_Transformer
from training.bc_trainer import BC_Trainer
from training.ppo_trainer import PPO_Trainer


def init():
    relabeler = Relabeler()
    collector = Collector()
    encoder = CLIP_Encoder()
    env = ViZDoom_Env()
    buffer = Buffer()
    policy_head = Policy_Head()
    temporal_transformer = Temporal_Transformer()
    bc_trainer = BC_Trainer()
    ppo_trainer = PPO_Trainer()

    relabeler.init()
    collector.init()
    encoder.init()
    env.init()
    buffer.init()
    policy_head.init()
    temporal_transformer.init()
    bc_trainer.init()
    ppo_trainer.init()

    encoder.test_clip()
    encoder.test_zero_shot()


def main():
    init()

    cfg = Config()
    cfg.init()

    if cfg.agent:
        print("Running Agent!")
        a = Agent(cfg.debug_agent, cfg.train_agent)
        a.run()

if __name__ == "__main__":
    main()
