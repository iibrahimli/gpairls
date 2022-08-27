"""
Train Webots agents.

Based on DBC code (https://github.com/facebookresearch/deep_bisim4control)
with modifications for MLP encoder, refactoring, custom evaluation, no CARLA
environment, custom stats logging, and no decoders.
"""

import time
from pathlib import Path
from datetime import datetime

import wandb
import torch
import numpy as np

import config, utils
from webots import RobotEnv
from ppr import PPR
from experts import ExpertPresets
from log import Logger
from agent import BisimAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def get_embedding(agent, obs):
    """
    Get embedding of observation.
    """
    if obs.ndim == 3:
        obs = np.expand_dims(obs, 0)
    with utils.eval_mode(agent.actor.encoder):
        emb = agent.actor.encoder(torch.tensor(obs).to(device)).detach().cpu().numpy()
    return emb


def get_action(env, agent, policy_reuse, obs, expert_config, epsilon):
    # try to get expert advice
    expert_action = env.get_expert_action(expert_config)
    emb = get_embedding(agent, obs)
    if expert_action is not None:
        action = expert_action
        if policy_reuse is not None:
            policy_reuse.add(emb, action)
        return action

    # try to get action from PPR
    if policy_reuse is not None:
        ppr_action, use_prob = policy_reuse.get(emb)
        # if can use ppr action, use it
        if ppr_action is not None and (np.random.rand() <= use_prob):
            return ppr_action

    # exploration
    if np.random.random() <= epsilon:
        return env.action_space.sample()

    # get action from agent
    with utils.eval_mode(agent):
        action = agent.sample_action(obs)

    return action


def evaluate(env, agent, L, step, n_episodes=5):
    """
    Evaluate agent on environment, averaged over n_episodes.
    """

    episode_rewards = []
    episode_lengths = []

    with utils.eval_mode(agent):
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                action = agent.sample_action(obs)
                next_obs, reward, done, _ = env.step(action)
                episode_reward += reward
                obs = next_obs
                episode_step += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_step)

    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)

    L.log("eval/episode_reward", mean_reward, step)
    L.log("eval/episode_length", mean_length, step)

    print(
        f"[EVAL] Step {step}: mean reward: {mean_reward:.5f}, mean length: {mean_length}"
    )

    return mean_reward, mean_length


def run_training(agent, env, policy_reuse, expert_config):

    agent.train()

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=config.REPLAY_BUFFER_CAPACITY,
        batch_size=config.BATCH_SIZE,
        device=device,
    )

    L = Logger(config.LOG_DIR, use_tb=False)

    epsilon = 0.2
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(config.TRAINING_STEPS):

        if done:
            if step > 0:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            L.log("train/episode_reward", episode_reward, step)

            wandb.log(
                {
                    "train": {
                        "episode_reward": episode_reward,
                        "epsilon": epsilon,
                    }
                },
                step=step,
            )

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log("train/episode", episode, step)


        # evaluate agent periodically
        if step % config.EVAL_FREQ == 0:
            L.log("eval/episode", episode, step)
            mean_reward, mean_length = evaluate(env, agent, L, step)
            agent.save(config.MODEL_DIR, step)
            # replay_buffer.save(buffer_dir)

            wandb.log(
                {
                    "eval": {
                        "episode_reward": mean_reward,
                        "episode_length": mean_length,
                    }
                },
                step=step,
            )

        # sample action for data collection
        if step < config.INIT_STEPS:
            action = env.action_space.sample()
        else:
            action = get_action(env, agent, policy_reuse, obs, expert_config, epsilon)

        # run training update
        if step >= config.INIT_STEPS:
            num_updates = 10 if step == config.INIT_STEPS else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        # allow infinite bootstrap
        done_float = 0.0 if episode_step + 1 >= env._max_episode_steps else float(done)
        episode_reward += reward

        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_float)

        obs = next_obs
        episode_step += 1
        epsilon *= 0.99
        if policy_reuse is not None:
            policy_reuse.step()
            wandb.log({"ppr_size": len(policy_reuse)}, step=step)


if __name__ == "__main__":

    utils.set_seed_everywhere(config.SEED)

    ENV_NAME = "RobotEnv-v0"
    env = RobotEnv()

    # create directories
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    agent = BisimAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_dim=config.HIDDEN_DIM,
        device=device,
    )

    # load model to continue training, if model file exists
    model_dir_path = Path("logs/model_cont/")
    if model_dir_path.exists() and any(model_dir_path.iterdir()):
        agent.load(model_dir_path, None)
        print("Loaded model to continue training")
    else:
        print("Training from scratch")

    expert_config = ExpertPresets.REALISTIC

    policy_reuse = PPR(init_prob=0.8, decay_rate=0.01)

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_NAME = f"{ENV_NAME}_bisim_{dt}"

    wandb_config = {
        "datetime": dt,
        "env": ENV_NAME,
        "agent": "bisim",
        "training_steps": config.TRAINING_STEPS,
        "eval_freq": config.EVAL_FREQ,
        "batch_size": config.BATCH_SIZE,
        "replay_buffer_capacity": config.REPLAY_BUFFER_CAPACITY,
        "init_steps": config.INIT_STEPS,
        "hidden_dim": config.HIDDEN_DIM,
        "encoder_dim": config.ENCODER_FEATURE_DIM,
        "actor_lr": config.ACTOR_LR,
        "critic_lr": config.CRITIC_LR,
    }

    # initialize wandb
    wandb_run_name = f"RobotEnv-bisim"
    if expert_config is not None:
        wandb_run_name += f"-{expert_config.name}"
    wandb.init(
        project="gpairls",
        entity="iibrahimli",
        name=wandb_run_name,
        config=wandb_config,
    )
    wandb.watch((agent.actor, agent.critic, agent.transition_model))

    # stop when ^C
    try:
        run_training(agent, env, policy_reuse, expert_config)
    except KeyboardInterrupt:
        print("Stopping early")
    finally:
        env.close()
        print("Done\n")