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
from training.bc_trainer import BC_Trainer
from training.ppo_trainer import PPO_Trainer


def main():
    """
    Main application loop.
    """
    cfg = Config()
    cfg.init()
    cfg.print_config()

    if cfg.config['encoder'] and cfg.config['test']:
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
    elif cfg.config['eval']:
        encoder = CLIP_Encoder(config=cfg.config)
        encoder.init()
        agent = Agent(config=cfg.config, encoder=encoder)
        agent.init()
        agent.load_checkpoint(cfg.config['policy'])
        agent.evaluate(episodes=cfg.config['episodes'])
        return
    elif cfg.config['export']:
        agent = Agent(config=cfg.config)
        agent.export_for_unity()
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
        encoder = CLIP_Encoder(config=cfg.config)
        encoder.init()
        ppo_trainer = PPO_Trainer(config=cfg.config, encoder=encoder)
        ppo_trainer.init()
        ppo_trainer.train(total_timesteps=cfg.config['timesteps'])
        return

    # Initialization
    collector = Collector(config=cfg.config)
    relabeler = Relabeler(config=cfg.config)
    encoder = CLIP_Encoder(config=cfg.config)

    collector.init()
    relabeler.init()
    encoder.init()

    bc_trainer = BC_Trainer(config=cfg.config, encoder=encoder)
    ppo_trainer = PPO_Trainer(config=cfg.config, encoder=encoder)
    agent = Agent(config=cfg.config, encoder=encoder)

    bc_trainer.init()
    ppo_trainer.init()
    agent.init()

    # 1. Collect demo trajectories
    collector.collect(episodes=cfg.config['episodes'])
    collector.inspect()

    # 2. Hindsight relabeling
    relabeler.relabel()
    relabeler.inspect()

    # 3. Behavior Cloning
    bc_trainer.train(epochs=cfg.config['epochs'])

    # 4. Proximal Policy Optimization
    ppo_trainer.train(total_timesteps=cfg.config['timesteps'])

    # 5. PPO Policy
    agent.load_checkpoint("ppo")
    agent.evaluate(episodes=cfg.config['episodes'])

    # 6. Behavior Cloning
    agent.load_checkpoint("bc")
    agent.evaluate(episodes=cfg.config['episodes'])

    # 7. Export Unity
    agent.export_for_unity()

if __name__ == "__main__":
    main()
