"""
Supervisor class for the epuck. Acts as an intermediary between the
gym environment and the experiment world in Webots.
"""

import numpy as np
from controller import Supervisor

from . import config


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
        self.rangefinder = self.robot.getDevice("rangefinder")

        # get max velocity
        self.max_velocity = self.left_motor.getMaxVelocity()

        # initialize camera
        self.camera.enable(timestep)

        # initialize range finder
        self.rangefinder.enable(timestep)
        self.rangefinder_max_range = self.rangefinder.getMaxRange()

        # initialize motors
        self._reset_motors()

        # [H, W, C], C = 4 for RGBD
        self.obs_shape = (self.camera.getHeight(), self.camera.getWidth(), 4)

        # positions
        self.init_robot_pos = self.robot.getSelf().getPosition()
        self.goal_pos = np.array(self.robot.getFromDef("goal").getPosition())

        # step once so that camera image is available
        self.step()

    def __del__(self):
        del self.robot

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
        Get the RGBD image from the camera in range [0, 255].

        Returns:
            numpy.ndarray: The image from the camera.
        """
        # RGB image, shape [H, W, C]
        img = np.array(self.camera.getImageArray())
        img = np.transpose(img, (2, 1, 0))

        # depth image, shape [H, W, 1]
        depth = np.frombuffer(
            self.rangefinder.getRangeImage(data_type="buffer"), dtype=np.float32
        )
        depth = np.reshape(depth, (1, *img.shape[:2]))
        depth = depth / self.rangefinder_max_range * 255
        depth = np.clip(depth, 0, 255).astype(np.uint8)

        # concatenate
        img = np.concatenate((img, depth), axis=0)

        return img

    def step(self):
        """
        Step the robot forward one timestep and return -1 if simulation ends.
        """

        return self.robot.step(self.timestep)

    def reset(self):
        """
        Reset the simulation.
        """

        # reset robot
        robot_node = self.robot.getSelf()
        robot_translation_field = robot_node.getField("translation")
        robot_translation_field.setSFVec3f(self.init_robot_pos)
        robot_node.resetPhysics()

        # reset goal
        goal_node = self.robot.getFromDef("goal")
        goal_translation_field = goal_node.getField("translation")
        goal_translation_field.setSFVec3f(self.goal_pos.tolist())
        goal_node.resetPhysics()

        self._reset_motors()
        self.step()

    def compute_distance_to_goal(self):
        """
        Compute the distance to the goal.

        Returns:
            float: The distance to the goal.
        """
        robot_pos = np.array(self.robot.getSelf().getPosition())
        dist = np.linalg.norm(robot_pos - self.goal_pos)
        return dist

    def _reset_motors(self):
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)