"""
OLD TRAINING SCRIPT

Train all agents for the paper.

Based on DBC code (https://github.com/facebookresearch/deep_bisim4control)
with modifications for MLP encoder, refactoring, custom evaluation, no CARLA
environment, custom stats logging, and no decoders.
"""

import time
from datetime import datetime

import gym
import torch
import numpy as np

import config
import utils
from webots import RobotEnv
from ppr import PPR
from log import Logger
from experts import MountainCarExpert
from agent import BaselineAgent, BisimAgent
from training_run import TrainingRun


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(env, agent, L, step, n_episodes=10):
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


def run_training(agent, env, eval_env, tr, policy_reuse, expert):

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=config.REPLAY_BUFFER_CAPACITY,
        batch_size=config.BATCH_SIZE,
        device=device,
    )

    L = Logger(config.LOG_DIR, use_tb=True)

    epsilon = 0.1
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(config.TRAINING_STEPS):
        if done:
            if step > 0:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % config.EVAL_FREQ == 0:
                L.log("eval/episode", episode, step)
                mean_reward, mean_length = evaluate(eval_env, agent, L, step)
                tr.add(episode=episode, mean_episode_reward=mean_reward, mean_episode_length=mean_length)
                agent.save(config.MODEL_DIR, step)
                # replay_buffer.save(buffer_dir)

            L.log("train/episode_reward", episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log("train/episode", episode, step)

        # sample action for data collection
        if step < config.INIT_STEPS:
            action = env.action_space.sample()
        else:
            # epsilon-greedy exploration
            if np.random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                # get state embedding
                with utils.eval_mode(agent.actor.encoder):
                    emb = agent.actor.encoder(torch.tensor(obs)).detach().cpu().numpy()

                # ask expert
                expert_action = expert.get_action(list(obs))
                if expert_action is not None:
                    action = expert_action
                    policy_reuse.add(emb, expert_action)
                else:
                    # try to get action from PPR
                    ppr_action, rate = policy_reuse.get(emb)
                
                    # if can use ppr action, use it
                    if ppr_action is not None and (np.random.rand() < rate):
                        action = ppr_action
                    else:
                        # get action from agent
                        with utils.eval_mode(agent):
                            action = agent.sample_action(obs)

        # run training update
        if step >= config.INIT_STEPS:
            num_updates = config.INIT_STEPS if step == config.INIT_STEPS else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        # allow infinite bootstrap
        done_float = 0.0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_float)

        obs = next_obs
        episode_step += 1
        epsilon *= 0.99
        policy_reuse.step()


if __name__ == "__main__":

    utils.set_seed_everywhere(config.SEED)

    ENV_NAME = "MountainCarContinuous-v0"

    # env = gym.make(ENV_NAME)
    env = RobotEnv()
    env.seed(config.SEED)

    eval_env = gym.make(ENV_NAME)
    eval_env.seed(config.SEED)

    # create directories
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    agent = BisimAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_dim=config.HIDDEN_DIM,
        device=device,
    )

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_NAME = f"{env.spec.id}_{agent.__class__.__name__}_{dt}"

    tr = TrainingRun(
        RUN_NAME,
        metadata={
            "datetime": dt,
            "env": env.spec.id,
            "agent": agent.__class__.__name__,
            "seed": config.SEED,
            "training_steps": config.TRAINING_STEPS,
            "eval_freq": config.EVAL_FREQ,
            "batch_size": config.BATCH_SIZE,
            "replay_buffer_capacity": config.REPLAY_BUFFER_CAPACITY,
            "init_steps": config.INIT_STEPS,
            "hidden_dim": config.HIDDEN_DIM,
            "actor_lr": config.ACTOR_LR,
            "critic_lr": config.CRITIC_LR,
        },
        tracked_stats=["episode", "mean_episode_reward", "mean_episode_length"],
    )

    policy_reuse = PPR(init_prob=0.8, decay_rate=0.05)

    expert = MountainCarExpert()

    # stop when ^C
    try:
        run_training(agent, env, eval_env, tr, policy_reuse, expert)
    except KeyboardInterrupt:
        print("Stopping early")
    finally:
        print("Saving training stats")
        tr.save(config.LOG_DIR / RUN_NAME)
        print("Done\n")
    
    print("Rendering trained agent in environment")
    obs = env.reset()
    done = False
    step = 0
    with utils.eval_mode(agent):
        while not done:
            env.render()
            action = agent.sample_action(obs)
            obs, reward, done, _ = env.step(action)
            print(f"step {step}: action = {action[0]:.3f}, reward = {reward:.3f}")
            step += 1