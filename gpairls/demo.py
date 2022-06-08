"""
Load an agent and run in the environment.
"""

import torch
import numpy as np

import config
from webots import RobotEnv
from agent import BisimAgent


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

    for episode in range(100):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, info = env.step(action)
            env.render(show_occupancy_grid=True)
            episode_reward += reward
        print(f"Episode {episode}   reward: {episode_reward:.3f}")