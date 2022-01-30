from sys import argv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


if len(argv) != 2:
    print("Usage: python3 plot_stats.py <path_to_csv_file>")
    exit(1)

stats_csv_path = argv[1].strip()
stats = pd.read_csv(stats_csv_path)

fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(6, 5))

ax0.plot(stats["episode"], stats["mean_episode_reward"])
ax0.set_title("Mean episode reward evaluated over 10 episodes")
ax0.set_xlabel("Episode")
ax0.set_ylabel("Mean episode reward")

ax1.plot(stats["episode"], stats["mean_episode_length"])
ax1.set_title("Mean episode length evaluated over 10 episodes")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Mean episode length")

plt.tight_layout()
plt.show()