# milliseconds
CONTROL_TIMESTEP = 100

# max steps per episode (seconds * seconds_per_step)
MAX_STEPS = round(10 * (1000 / CONTROL_TIMESTEP))

# reward for not reaching the goal
STEP_REWARD = -1.

# reward for reaching the goal
GOAL_REWARD = 100.0

# distance (m) under which the goal is considered reached
GOAL_DISTANCE_THRESHOLD = 0.1