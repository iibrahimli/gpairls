"""
Webots robot environment config
"""
from pathlib import Path


# milliseconds
CONTROL_TIMESTEP = 500

# max steps per episode (minutes * 60 * seconds_per_step)
MAX_TIME_MINUTES = 15
MAX_STEPS = round(MAX_TIME_MINUTES * 60 * (1000 / CONTROL_TIMESTEP))

# reward for not reaching the goal
STEP_REWARD = -0.01

# reward for reaching the goal
GOAL_REWARD = 10.

# reward for collision
COLLISION_REWARD = -0.1

# distance (meters) under which the goal is considered reached
GOAL_DISTANCE_THRESHOLD = 0.1

# occupancy grid resolution (meters)
OCCUPANCY_GRID_RESOLUTION = 0.02

# path to the occupancy grid
OCCUPANCY_GRID_PATH = Path(__file__).parent.resolve() / "occupancy_grid.npy"