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
        
        # continuous action space in range [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # RGB image observation
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=config.CAM_SHAPE, dtype=np.uint8
        )
        
        # robot stuff
        self.controller = EpuckSupervisor(config.CONTROL_TIMESTEP)

        # plt AxisImage to render the image
        self.ax_im = None

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset the environment.
        """
        super().reset(seed=seed)

        # TODO: add supervisor stuff to reset env

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

        if self.controller.step() != -1:
            self.controller.move(action)
        else:
            done = True

        reward = -0.1
        observation = self._get_obs()

        if return_info:
            return (observation, reward, done, self._get_info())
        else:
            return (observation, reward, done)

    def render(self):
        """
        Render the environment.
        """
        if self.ax_im is None:
            # create figure for camera
            ax1 = plt.subplot(111)
            ax1.set_title("Camera image")
            self.ax_im = ax1.imshow([[0]])
            plt.ion()
        self.ax_im.set_data(self._get_obs())
        plt.pause(config.CONTROL_TIMESTEP / 1000)
    
    def close(self):
        self.ax_im = None
        plt.close("all")

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
        # TODO: get info such as distance to target
        return None
