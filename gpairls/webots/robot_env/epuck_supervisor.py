"""
Supervisor class for the epuck. Acts as an intermediary between the
gym environment and the experiment world in Webots.
"""

import sys
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage.transform import resize
from loguru import logger

sys.path.insert(0, "/Applications/Webots.app/lib/controller/python39")  # local
sys.path.insert(0, "/usr/local/webots/lib/controller/python38")  # server
from controller import Supervisor  # type: ignore

from gpairls.webots.robot_env import config, utils


class EpuckSupervisor:
    def __init__(self, timestep):
        self.robot = Supervisor()
        self.timestep = timestep
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.camera = self.robot.getDevice("camera")
        self.rangefinder = self.robot.getDevice("rangefinder")
        self.touch_sensor = self.robot.getDevice("touch sensor")

        # get max velocity
        self.max_velocity = self.left_motor.getMaxVelocity()

        # initialize camera
        self.camera.enable(timestep)

        # initialize range finder
        self.rangefinder.enable(timestep)
        self.rangefinder_max_range = self.rangefinder.getMaxRange()

        # initialize motors
        self._reset_motors()

        # initialize touch sensor
        self.touch_sensor.enable(timestep)

        # [C, H, W], C = 4 for RGBD
        self.obs_shape = (4, self.camera.getHeight(), self.camera.getWidth())

        # poses
        self.init_robot_pos = self.robot.getSelf().getPosition()
        self.init_robot_ori = [0, 0, 1, 1.57]  # hardcode orientation
        # self.init_robot_ori = self.robot.getSelf().getOrientation()
        self.goal_pos = self.robot.getFromDef("goal").getPosition()
        self.goal_ori = self.robot.getFromDef("goal").getOrientation()
        # self.init_robot_pos[2] += 0.005

        # occupancy grid
        self.arena_size = (
            self.robot.getFromDef("arena").getField("floorSize").getSFVec2f()
        )

        self.occupancy_grid = None
        self.advice_grid = None
        self._ensure_expert_data()

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
        cur_vl = self.left_motor.getVelocity()
        cur_vr = self.right_motor.getVelocity()
        vl = vr = self.max_velocity

        if direction < 0:
            vl *= 1 - abs(direction)
        elif direction > 0:
            vr *= 1 - abs(direction)

        # smooth v
        ALPHA = 0.9
        vl = np.clip(ALPHA * vl + (1 - ALPHA) * cur_vl, 0, self.max_velocity)
        vr = np.clip(ALPHA * vr + (1 - ALPHA) * cur_vr, 0, self.max_velocity)

        self.left_motor.setVelocity(vl)
        self.right_motor.setVelocity(vr)

    def get_cam_image(self):
        """
        Get the RGBD image from the camera in range [0, 255].

        Returns:
            numpy.ndarray: The image from the camera.
        """
        # RGB image, shape [C, H, W]
        img = np.array(self.camera.getImageArray())
        img = np.transpose(img, (2, 1, 0))

        # depth image, shape [1, H, W]
        depth = np.frombuffer(
            self.rangefinder.getRangeImage(data_type="buffer"), dtype=np.float32
        )
        depth = np.reshape(depth, (1, *img.shape[1:]))
        depth = depth / self.rangefinder_max_range * 255
        depth = np.clip(depth, 0, 255).astype(np.uint8)

        # concatenate
        img = np.concatenate((img, depth), axis=0).astype(np.uint8)

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
        self._move_node(robot_node, self.init_robot_pos, ori=self.init_robot_ori)

        # reset goal
        goal_node = self.robot.getFromDef("goal")
        self._move_node(goal_node, self.goal_pos, ori=self.goal_ori)

        self._reset_motors()
        self.step()

    def compute_distance_to_goal(self) -> float:
        """
        Compute the distance to the goal.

        Returns:
            float: The distance to the goal.
        """
        robot_pos = np.array(self.robot.getSelf().getPosition())
        dist = np.linalg.norm(robot_pos - np.array(self.goal_pos))
        return dist

    def get_expert_action(self, expert_config):
        """
        Choose an expert action for current state of the simulation (not
        exactly observation). The best action i.e. direction is assumed to be
        the one on the shortest path to the goal.

        Returns:
            float: The direction to move in, in range [-1, 1], or None
            if the robot is at the goal or the goal is unreachable from robot's
            position (the latter shouldn't happen).
        """
        if np.random.uniform() > expert_config.availability:
            return None

        if np.random.uniform() > expert_config.accuracy:
            return np.random.uniform(-1, 1)

        robot_pos = tuple(self.robot.getSelf().getPosition())[:2]
        robot_cell = self._world_to_grid_coords(*robot_pos)
        robot_orientation = self.robot.getSelf().getField("rotation").getSFRotation()
        robot_angle = robot_orientation[-1]  # radians, x-axis is "down"

        # compute optimal direction
        advice_angle = self.advice_grid[robot_cell]

        # robot_angle and next_angle are in range [-pi, pi]
        # therefore, the angle difference is in range [-2pi, 2pi]
        # need to normalize it to [-1, 1]
        angle_diff = advice_angle - robot_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # normalize angle difference to [-1, 1]
        angle_diff /= math.pi

        if np.abs(angle_diff) < 0.5:
            return angle_diff
        else:
            # angle too large - turn as much as you can
            return np.sign(angle_diff)


    def is_collided(self) -> bool:
        """
        Check if the robot is collided.

        Returns:
            bool: True if the robot is collided, False otherwise.
        """
        return bool(self.touch_sensor.getValue())

    def step_back(self, dist=0.07):
        """
        Move the robot back in the opposite direction of current rotation
        (usually after a collision)
        """
        robot_pos = np.array(self.robot.getSelf().getPosition())
        robot_rotation = self.robot.getSelf().getField("rotation").getSFRotation()

        # compute opposite direction
        angle = (robot_rotation[-1] + np.pi) % (2 * np.pi)
        if robot_rotation[2] < 0:
            angle = 2 * np.pi - angle

        # compute new position
        new_pos = robot_pos
        new_pos[:2] += dist * np.array([np.cos(angle), np.sin(angle)])

        # clip if the position is outside arena
        arena_x, arena_y = self.arena_size
        buffer = 0.08
        new_pos[0] = np.clip(new_pos[0], -arena_x / 2 + buffer, arena_x / 2 - buffer)
        new_pos[1] = np.clip(new_pos[1], -arena_y / 2 + buffer, arena_y / 2 - buffer)

        # move to new position
        self._move_node(self.robot.getSelf(), new_pos.tolist())
        self._reset_motors()

    def render_occupancy_grid(self) -> np.ndarray:
        """
        Render the occupancy grid.
        """

        def _mark_position(grid, x, y, size, value):
            grid[x : x + size, y : y + size] = value

        # mark positions of the robot and the goal
        robot_grid_pos = self._world_to_grid_coords(
            *tuple(self.robot.getSelf().getPosition())[:2]
        )
        goal_grid_pos = self._world_to_grid_coords(*tuple(self.goal_pos)[:2])
        occ_grid_img = self.occupancy_grid.copy()
        _mark_position(occ_grid_img, *robot_grid_pos, 5, 2)
        _mark_position(occ_grid_img, *goal_grid_pos, 5, 3)

        # shortest_path = self.get_shortest_path(
        #     tuple(self.robot.getSelf().getPosition())[:2]
        # )
        # for grid_pos in shortest_path:
        #     _mark_position(occ_grid_img, *grid_pos, 2, 4)

        # convert to RGB image
        occ_grid_img = np.array(
            [
                # 0: white background
                [255, 255, 255],
                # 1: black obstacles
                [0, 0, 0],
                # 2: red robot
                [255, 0, 0],
                # 3: green goal
                [0, 255, 0],
                # 4: blue shortest path
                [0, 0, 255],
            ]
        )[occ_grid_img]

        # resize
        occ_grid_img = resize(occ_grid_img / 255.0, (256, 512))

        return occ_grid_img

    def compute_occupancy_grid(self) -> np.ndarray:
        """
        Compute the occupancy grid of the simulation arena. This implementation
        is slow and advances the simulation, therefore is only used once during
        initialization of the environment object (not on .reset()).
        """
        occ_res = config.OCCUPANCY_GRID_RESOLUTION
        obj_z = 0.02

        arena_x, arena_y = self.arena_size
        occ_grid = np.zeros(
            (round(arena_x / occ_res), round(arena_y / occ_res)), dtype=np.int8
        )

        # get robot node
        robot_node = self.robot.getSelf()

        # temporarily move the robot and goal out of the arena
        goal_node = self.robot.getFromDef("goal")
        self._move_node(goal_node, [arena_x, arena_y, 0])

        # move touch sensor (and the robot to which it is attached) to the
        # center of each grid cell to check for collisions

        self.robot.step(1)

        for i in range(occ_grid.shape[0]):
            for j in range(occ_grid.shape[1]):

                # move obj to center of grid cell
                x, y = self._grid_to_world_coords(i, j)
                self._move_node(robot_node, [x, y, obj_z])

                # check for collisions
                occ_grid[i, j] = np.nan_to_num(self.touch_sensor.getValue())

                self.robot.step(1)

        # move robot and goal back to their original position
        self.reset()
        self.robot.step(1)
        np.save(config.OCCUPANCY_GRID_PATH, occ_grid)
        return occ_grid

    def compute_advice_grid(self) -> np.ndarray:
        """
        Precompute the expert advice grid. The grid has shape [H, W] and
        where each cell represents the optimal direction to take at each cell
        """
        goal_cell = self._world_to_grid_coords(*tuple(self.goal_pos)[:2])
        advice_grid = utils.compute_advice_grid(self.occupancy_grid, goal_cell)
        np.save(config.ADVICE_GRID_PATH, advice_grid)
        return advice_grid

    def _ensure_expert_data(self):
        """
        Load stuff for expert model, precompute if necessary
        Sets:
         - self.occupancy_grid
         - self.advice_grid
        """
        if Path(config.OCCUPANCY_GRID_PATH).is_file():
            self.occupancy_grid = np.load(config.OCCUPANCY_GRID_PATH)
            logger.info(f"Loaded occupancy grid from {config.OCCUPANCY_GRID_PATH}")
        else:
            self.occupancy_grid = self.compute_occupancy_grid()
            logger.info(f"Computed occupancy grid")

        if Path(config.ADVICE_GRID_PATH).is_file():
            self.advice_grid = np.load(config.ADVICE_GRID_PATH)
            logger.info(f"Loaded advice grid from {config.ADVICE_GRID_PATH}")
        else:
            self.advice_grid = self.compute_advice_grid()
            logger.info(f"Computed advice grid")

    def _world_to_grid_coords(self, x, y) -> Tuple[float, float]:
        """
        Convert world coordinates to grid coordinates.
        """
        grid_res = config.OCCUPANCY_GRID_RESOLUTION
        grid_x = np.floor((x + self.arena_size[0] / 2) / grid_res)
        grid_y = np.floor((y + self.arena_size[1] / 2) / grid_res)
        return round(grid_x), round(grid_y)

    def _grid_to_world_coords(self, grid_x, grid_y) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates (center of grid cell).
        """
        grid_res = config.OCCUPANCY_GRID_RESOLUTION
        x = grid_x * grid_res - self.arena_size[0] / 2 + grid_res / 2
        y = grid_y * grid_res - self.arena_size[1] / 2 + grid_res / 2
        return x, y

    def _move_node(self, node, pos, ori=None, reset_physics=True):
        """
        Move the given node to the given position & orientation.
        """
        node.getField("translation").setSFVec3f(pos)
        if ori is not None:
            node.getField("rotation").setSFRotation(ori)
        if reset_physics:
            node.resetPhysics()

    def _reset_motors(self):
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
