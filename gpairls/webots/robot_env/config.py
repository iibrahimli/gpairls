"""
Webots robot environment config
"""
from pathlib import Path


# milliseconds
CONTROL_TIMESTEP = 100

# max steps per episode (minutes * 60 * seconds_per_step)
MAX_STEPS = round(10 * 60 * (1000 / CONTROL_TIMESTEP))

# reward for not reaching the goal
STEP_REWARD = -1.

# reward for reaching the goal
GOAL_REWARD = 1000.0

# reward for collision
COLLISION_REWARD = -100.0

# distance (meters) under which the goal is considered reached
GOAL_DISTANCE_THRESHOLD = 0.1

# occupancy grid resolution (meters)
OCCUPANCY_GRID_RESOLUTION = 0.02

# path to the occupancy grid
OCCUPANCY_GRID_PATH = Path(__file__).parent.resolve() / "occupancy_grid.npy"