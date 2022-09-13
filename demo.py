"""
Load an agent and run in the environment.
"""

import argparse
import pathlib

import torch
from loguru import logger

from gpairls import config, utils
from gpairls.experts import ExpertConfig
from gpairls.webots import RobotEnv
from gpairls.agent import BisimAgent


# enable colors for logger 
logger = logger.opt(colors=True)

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

    mconf = utils.load_model_config(
        args.checkpoint / config.MODEL_CONFIG_PATH.name
    )

    agent = BisimAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_dim=mconf["hidden_dim"],
        encoder_feature_dim=mconf["encoder_feature_dim"],
        encoder_num_layers=mconf["encoder_num_layers"],
        encoder_num_filters=mconf["encoder_num_filters"],
        decoder_dim=mconf["decoder_dim"],
        transition_model_dim=mconf["transition_model_dim"],
        device=device,
    )

    expert_config = ExpertConfig(1, 1)

    agent.load(args.checkpoint, None)
    logger.info(f"Loaded model from {args.checkpoint}")

    for episode in range(10):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done:
            action = agent.sample_action(obs)
            # action = [env.get_expert_action(expert_config)]

            obs, reward, done, info = env.step(action)
            logger.info(f"<c>step {step}</c> - action: {action[0]:.4f} - reward: {reward:.4f}")

            episode_reward += reward
            step += 1
        
        logger.info(f"<c><i>Episode {episode}</i></c>  reward: <r>{episode_reward:.3f}</r>")
