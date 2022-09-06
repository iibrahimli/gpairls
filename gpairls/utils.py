"""
Utility functions.

Based on DBC code (https://github.com/facebookresearch/deep_bisim4control)
"""

import os
import copy
import random
import yaml
from typing import Dict

import torch
import numpy as np

from gpairls import config


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class eval_mode:
    """
    Put model(s) in evaluation mode & restore previous state afterwards.
    """

    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return (
                obses,
                actions,
                rewards,
                next_obses,
                not_dones,
                torch.as_tensor(self.k_obses[idxs], device=self.device),
            )
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.curr_rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end


def ensure_obs_dims(obs):
    if obs.ndim == 3:
        obs = np.expand_dims(obs, 0)
    return obs


def get_embedding(agent, obs, device):
    """
    Get embedding of observation.
    """
    obs = ensure_obs_dims(obs)
    with eval_mode(agent.actor.encoder):
        emb = agent.actor.encoder(torch.tensor(obs).to(device)).detach().cpu().numpy()
    return emb


def get_trajectory(agent, env, device):
    """Get trajectory from agent"""
    obss = []
    occ_grids = []
    embs = []
    actions = []
    rewards = []

    obs = env.reset()
    done = False
    while not done:
        action = agent.sample_action(obs)
        next_obs, reward, done, _ = env.step(action)
        occ_grid = env.get_occupancy_grid_image()

        # append
        obss.append(obs)
        occ_grids.append(occ_grid)
        embs.append(get_embedding(agent, obs, device))
        actions.append(action)
        rewards.append(reward)

        obs = next_obs

    traj = {
        "obs": np.array(obss),
        "embs": np.array(embs),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
    }

    return traj


def save_trajectory(
    path: str,
    obs: np.ndarray,
    emb: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
):
    """
    Save agent trajectory.

    Args:
        path: Path to the file
        obs: Array of observations, shape [N, C, H, W] where N is number of steps
        emb: Array of embeddings, shape [N, D], where D is emb dimension
        actions: Array of actions, shape [N, 1]
        rewards: Array of rewards, shape [N, 1]
    """
    np.savez_compressed(
        path,
        obs=obs,
        emb=emb,
        actions=actions,
        rewards=rewards,
    )


def load_trajectory(path: str) -> Dict[str, np.ndarray]:
    """Load trajectory saved by save_trajectory"""
    return np.load(path)


_MODEL_CONFIG_KEYS = (
    "ENCODER_FEATURE_DIM",
    "ENCODER_NUM_LAYERS",
    "ENCODER_NUM_FILTERS",
    "HIDDEN_DIM",
    "DECODER_DIM",
    "TRANSITION_MODEL_DIM",
)


def save_model_config(path=None):
    """Save model stuff to a YAML file for loading later"""
    model_config = {}
    for k in _MODEL_CONFIG_KEYS:
        model_config[k.lower()] = getattr(config, k)
    path = path or config.MODEL_CONFIG_PATH
    with open(path, "w") as f:
        yaml.dump(model_config, f)


def load_model_config(path=None):
    """Load model config from YAML"""
    path = path or config.MODEL_CONFIG_PATH
    with open(path, "r") as f:
        model_config = yaml.safe_load(f)
    return model_config


def param_count(model):
    return sum(param.numel() for param in model.parameters())