"""
OpenAI Gym environment.
"""

from typing import Iterable
import gym
import numpy as np
import matplotlib.pyplot as plt

from . import config
from .epuck_supervisor import EpuckSupervisor


class RobotEnv(gym.Env):
    """
    Environment for the Webots robotic navigation task. Acts
    as an extern controller for the robot using EpuckSupervisor class,
    therefore requires starting Webots simulation separately.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Initialize the environment.
        """
        super(RobotEnv, self).__init__()

        # robot stuff
        self.controller = EpuckSupervisor(config.CONTROL_TIMESTEP)

        # continuous action space in range [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # RGB image observation
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.controller.obs_shape, dtype=np.uint8
        )

        # maximum steps per episode
        self.max_steps = config.MAX_STEPS
        self._max_episode_steps = self.max_steps

        # current env step
        self.step_count = 0

        # plt AxisImages to render the observation
        self.rgb_axis_image = None
        self.depth_axis_image = None

    def reset(self, seed=None):
        """
        Reset the environment.
        """
        super().reset(seed=seed)
        self.step_count = 0

        self.controller.reset()

        observation = self._get_obs()
        return observation

    def step(self, action):
        """
        Perform one action in the environment.
        """
        done = False
        reward = config.STEP_REWARD
        self.step_count += 1

        # move in simulation & check if sim ended
        if self.controller.step() != -1:
            if isinstance(action, Iterable):
                action = action[0]
            self.controller.move(action)
        else:
            done = True

        # check if max steps reached
        if self.step_count == self.max_steps:
            done = True

        # check collision
        if not done and self.controller.is_collided():
            reward = config.COLLISION_REWARD
            done = True

        # check if goal is reached
        if not done:
            goal_dist = self.controller.compute_distance_to_goal()
            if goal_dist < config.GOAL_DISTANCE_THRESHOLD:
                done = True
                reward = config.GOAL_REWARD

        observation = self._get_obs()

        return (observation, reward, done, self._get_info())

    def render(self, show_occupancy_grid=False):
        """
        Render the environment.
        """
        if self.rgb_axis_image is None:
            # create figure for observation
            n_cols = 2 + int(show_occupancy_grid)
            fig, axes = plt.subplots(
                1,
                n_cols,
                figsize=(3 * n_cols, 3),
                gridspec_kw={"width_ratios": [1, 1, 2][:n_cols]},
            )
            fig.suptitle("Observation")

            axes[0].set_title("RGB image")
            self.rgb_axis_image = axes[0].imshow([[0]], vmin=0, vmax=255)

            axes[1].set_title("Depth image")
            self.depth_axis_image = axes[1].imshow(
                [[0]], cmap="gray_r", vmin=0, vmax=255
            )

            if show_occupancy_grid:
                axes[2].set_title("Occupancy grid")
                self.occupancy_axis_image = axes[2].imshow([[0, 0]], vmin=0, vmax=255)

            plt.ion()
        obs = self._get_obs()
        self.rgb_axis_image.set_data(obs[:3, ...].transpose(1, 2, 0))
        self.depth_axis_image.set_data(obs[3])
        if show_occupancy_grid:
            self.occupancy_axis_image.set_data(self.controller.render_occupancy_grid())
        plt.pause(config.CONTROL_TIMESTEP / 1000)

    def close(self):
        self.ax_im = None
        plt.close("all")
        del self.controller
    
    def get_expert_action(self):
        """
        Get the expert action from current state.
        """
        return self.controller.get_expert_action()

    def _get_obs(self):
        """
        Get the observation from current state.
        """
        img = self.controller.get_cam_image()
        return img

    def _get_info(self):
        """
        Get the info from current state.
        """
        return None
