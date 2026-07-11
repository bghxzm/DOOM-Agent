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
from encoder.og_clip_encoder_test import OG_CLIP_Encoder_Test
from environment.vizdoom_env import ViZDoom_Env
from memory.buffer import Buffer
from model.policy_head import Policy_Head
from model.temporal_transformer import Temporal_Transformer
from training.bc_trainer import BC_Trainer
from training.ppo_trainer import PPO_Trainer


def main():
    """
    Main application loop.
    """
    cfg = Config()
    cfg.init()
    cfg.print_config()

    if cfg.config['agent']:
        a = Agent()
        a.init(cfg.agent)
        return
    elif cfg.config['encoder'] and cfg.config['test']:
        encoder = OG_CLIP_Encoder_Test(config=cfg.config)
        encoder.init()
        encoder.run_open_clip()
        encoder.test_clip()
        encoder.test_zero_shot()
        return
    elif cfg.config['environment'] and cfg.config['test']:
        env = ViZDoom_Env(config=cfg.config)
        env.init()
        env.run_default_scenario()
        env.test_basic_loop()
        return
    elif cfg.config['collect']:
        collector = Collector(config=cfg.config)
        collector.init()
        collector.collect(episodes=cfg.config['episodes'])
        collector.inspect()
        return
    elif cfg.config['relabel']:
        relabeler = Relabeler(config=cfg.config)
        relabeler.init()
        relabeler.relabel()
        relabeler.inspect()
        return
    elif cfg.config['bc']:
        encoder = CLIP_Encoder(config=cfg.config)
        encoder.init()
        bc_trainer = BC_Trainer(config=cfg.config, encoder=encoder)
        bc_trainer.init()
        bc_trainer.train(epochs=cfg.config['epochs'])
        return
    elif cfg.config['ppo']:
        ppo_trainer = PPO_Trainer()
        ppo_trainer.init()
        return

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
