"""
Standalone script to precompute the expert advice
"""

from gpairls.webots.robot_env import config

try:
    from gpairls.webots.robot_env.epuck_supervisor import EpuckSupervisor
except ImportError as e:
    raise ImportError(
        "Can't import EpuckSupervisor. Probably it's an issue with the "
        "path and the Webots controller can't be found. Edit the file "
        "epuck_supervisor.py & set the appropriate paths."
    ) from e


es = EpuckSupervisor(config.CONTROL_TIMESTEP)
print("Grid shape:", es.occupancy_grid.shape)
print("Saved occupancy grid to ", config.OCCUPANCY_GRID_PATH)
print("Saved advice grid to ", config.ADVICE_GRID_PATH)