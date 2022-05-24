"""
Supervisor class for the epuck. Acts as an intermediary between the
gym environment and the experiment world in Webots.
"""

import numpy as np
from controller import Supervisor


# TODO:
# - reset experiment
# - compute distance to goal
# - compute reward


class EpuckSupervisor:
    def __init__(self, timestep):
        self.robot = Supervisor()
        self.timestep = timestep
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.camera = self.robot.getDevice("camera")

        # get max velocity
        self.max_velocity = self.left_motor.getMaxVelocity()

        # initialize camera
        self.camera.enable(timestep)

        # initialize motors
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # step once so that camera image is available
        self.step()

    def move(self, direction):
        """
        Move the robot at maximum speed in the given direction.

        Args:
            robot (Robot): The robot to control.
            direction (float): The direction to move in, a degree in range [-1, 1]
                where -1 is left and 1 is right and 0 is forward.
        """

        v_l = v_r = self.max_velocity
        if direction < 0:
            v_l *= 1 - abs(direction)
        elif direction > 0:
            v_r *= 1 - abs(direction)

        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

    def get_cam_image(self):
        """
        Get the image from the camera.

        Returns:
            numpy.ndarray: The image from the camera.
        """

        img = np.array(self.camera.getImageArray())
        img = np.transpose(img, (1, 0, 2))
        return img
    
    def step(self):
        """
        Step the robot forward one timestep and return -1 if simulation ends.
        """

        return self.robot.step(self.timestep)