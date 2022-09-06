"""
Load an agent and run in the environment.
"""

import argparse
import pathlib

import torch

from gpairls import config, utils
from gpairls.webots import RobotEnv
from gpairls.agent import BisimAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        metavar="PATH",
        required=True,
        type=pathlib.Path,
        help="Path to the model directory",
    )
    args = ap.parse_args()

    env = RobotEnv()

    model_config = utils.load_model_config(
        args.checkpoint / config.MODEL_CONFIG_PATH.name
    )
    config = utils.patch_config_with_model_config(config, model_config)

    agent = BisimAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_dim=config.HIDDEN_DIM,
        device=device,
    )

    agent.load(args.checkpoint, None)
    print("Loaded model from", args.checkpoint)

    for episode in range(10):
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
        print()
        print(f"Episode {episode}   reward: {episode_reward:.3f}")
        print()
