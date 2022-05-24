import numpy as np

from webots import RobotEnv

# TODO add depth channel to camera

env = RobotEnv()
print("Initialized environment.")

obs = env.reset()
done = False
while not done:
    action = np.random.uniform(-0.1, 0.1)
    obs, reward, done = env.step(action)
    print(f"Reward: {reward}")
    env.render()

print("Simulation finished.")