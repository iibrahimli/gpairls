"""
Load an agent and run in the environment.
"""

import torch
import numpy as np

from gpairls import config
from gpairls.webots import RobotEnv
from gpairls.agent import BisimAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    env = RobotEnv()

    agent = BisimAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_dim=config.HIDDEN_DIM,
        device=device,
    )

    checkpoint_dir = "logs/model"

    agent.load(checkpoint_dir, 0)
    print("Loaded model from", checkpoint_dir)

    for episode in range(100):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, info = env.step(action)
            print(f"step {step} - action: {action[0]:.4f} - reward: {reward:.4f}")
            # env.render(show_occupancy_grid=True)
            episode_reward += reward
            step += 1
        print(f"Episode {episode}   reward: {episode_reward:.3f}")