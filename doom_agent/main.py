"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

main.py
"""

from config import Config
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

    policy_head = Policy_Head()
    policy_head.init()

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
    cfg.print_config()

    if cfg.config['agent']:
        a = Agent()
        a.init(cfg.agent)
    elif cfg.config['encoder'] and cfg.config['test']:
        encoder = CLIP_Encoder(config=cfg.config)
        encoder.init()
        encoder.run_open_clip()
        encoder.test_clip()
        encoder.test_zero_shot()
    elif cfg.config['environment'] and cfg.config['test']:
        env = ViZDoom_Env(config=cfg.config)
        env.init()
        env.run_default_scenario()
        env.test_basic_loop()

    encoder = CLIP_Encoder(config=cfg.config)
    encoder.init()

    buffer = Buffer()
    buffer.init()

    transformer = Temporal_Transformer()
    transformer.to(cfg.config['device'])
    transformer.init()

    env = ViZDoom_Env(config=cfg.config, encoder=encoder, buffer=buffer,
                      transformer=transformer)
    env.init()
    env.run_default_scenario()

if __name__ == "__main__":
    main()
