"""
OpenAI Gym environment.
"""

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

        # current env step
        self.step_count = 0

        # plt AxisImages to render the observation
        self.rgb_axis_image = None
        self.depth_axis_image = None

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset the environment.
        """
        super().reset(seed=seed)
        self.step_count = 0

        self.controller.reset()

        observation = self._get_obs()
        if return_info:
            return (observation, self._get_info())
        else:
            return observation

    def step(self, action, return_info=False):
        """
        Perform one action in the environment.
        """

        done = False
        reward = config.STEP_REWARD
        self.step_count += 1

        # move in simulation & check if sim ended
        if self.controller.step() != -1:
            self.controller.move(action)
        else:
            done = True

        # check if max steps reached
        if self.step_count == self.max_steps:
            done = True

        # check if goal is reached
        goal_dist = self.controller.compute_distance_to_goal()
        if goal_dist < config.GOAL_DISTANCE_THRESHOLD:
            done = True
            reward = config.GOAL_REWARD

        observation = self._get_obs()

        if return_info:
            return (observation, reward, done, self._get_info())
        else:
            return (observation, reward, done)

    def render(self):
        """
        Render the environment.
        """
        if self.rgb_axis_image is None:
            # create figure for observation
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            fig.suptitle("Observation")
            axes[0].set_title("RGB image")
            axes[1].set_title("Depth image")
            self.rgb_axis_image = axes[0].imshow([[0]], vmin=0, vmax=255)
            self.depth_axis_image = axes[1].imshow(
                [[0]], cmap="gray_r", vmin=0, vmax=255
            )
            plt.ion()
        obs = self._get_obs()
        self.rgb_axis_image.set_data(obs[:3, ...].transpose(1, 2, 0))
        self.depth_axis_image.set_data(obs[3])
        plt.pause(config.CONTROL_TIMESTEP / 1000)

    def close(self):
        self.ax_im = None
        plt.close("all")
        del self.controller

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
