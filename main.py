import numpy as np

from gpairls.webots import RobotEnv


env = RobotEnv()
print("Initialized environment.")

for episode in range(1):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # action = np.random.uniform(-0.1, 0.1)
        action = env.get_expert_action()
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        # env.render(show_occupancy_grid=True)

    print(f"[{episode}] Episode reward: {episode_reward}")