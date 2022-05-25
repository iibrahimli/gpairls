import numpy as np

from gpairls.webots import RobotEnv


env = RobotEnv()
print("Initialized environment.")

for episode in range(50):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = np.random.uniform(-0.1, 0.1)
        obs, reward, done = env.step(action)
        episode_reward += reward
        # env.render()