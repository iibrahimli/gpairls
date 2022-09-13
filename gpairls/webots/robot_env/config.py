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
STEP_REWARD = 0.

# reward for reaching the goal
GOAL_REWARD = 100.

# reward for collision
COLLISION_REWARD = -1.

# distance (meters) under which the goal is considered reached
GOAL_DISTANCE_THRESHOLD = 0.2

# occupancy grid resolution (meters)
OCCUPANCY_GRID_RESOLUTION = 0.01

WEBOTS_DATA_DIR = Path(__file__).parent.resolve() / "data"
WEBOTS_DATA_DIR.mkdir(exist_ok=True)

# path to the occupancy grid
OCCUPANCY_GRID_PATH = WEBOTS_DATA_DIR / "occupancy_grid.npy"

# path to the advice grid
ADVICE_GRID_PATH = WEBOTS_DATA_DIR / "advice_grid.npy"