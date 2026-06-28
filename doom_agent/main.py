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
    """
    Initialize every package.
    """
    relabeler = Relabeler()
    relabeler.init()

    collector = Collector()
    collector.init()

    buffer = Buffer()
    buffer.init()

    policy_head = Policy_Head()
    policy_head.init()

    temporal_transformer = Temporal_Transformer()
    temporal_transformer.init()

    bc_trainer = BC_Trainer()
    bc_trainer.init()

    ppo_trainer = PPO_Trainer()
    ppo_trainer.init()


def main():
    """
    Main application loop.
    """
    init()

    cfg = Config()
    cfg.init()

    if cfg.agent:
        a = Agent()
        a.init(cfg.agent)
    elif cfg.encoder:
        encoder = CLIP_Encoder()
        encoder.init()
        encoder.test_clip()
        encoder.test_zero_shot()
    elif cfg.environment:
        env = ViZDoom_Env()
        env.init()
        env.run_default_scenario()
        env.test_basic_loop()


if __name__ == "__main__":
    main()
