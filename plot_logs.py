import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the output folder exists
os.makedirs("logs/images", exist_ok=True)

# Load data
df = pd.read_csv("logs/DQN_training_log.csv")
x = df["total_steps"]

# 1) Plot Mean Final Distance + Mean Reward
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(x, df["mean_eval_final_dist"], color="tab:blue", marker="o", label="Mean Eval Dist")
ax1.plot(x, df["mean_reward_100_eps"], color="tab:green", marker="x", label="Mean Reward (100 eps)")
ax1.set_xlabel("Total Training Steps")
ax1.set_ylabel("Distance (m) / Reward", color="black")
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(loc="upper right")
plt.title("Distance & Reward over Training")
plt.tight_layout()
fig.savefig("logs/images/distance_reward.png", dpi=300)
plt.close(fig)

# 2) Plot Epsilon Schedule
fig, ax2 = plt.subplots(figsize=(8,5))
ax2.plot(x, df["epsilon"], color="tab:red", linestyle="--", label="ε (exploration)")
ax2.set_xlabel("Total Training Steps")
ax2.set_ylabel("Epsilon", color="tab:red")
ax2.set_ylim(0,1)
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(loc="upper right")
plt.title("Exploration Rate over Training")
plt.tight_layout()
fig.savefig("logs/images/epsilon_schedule.png", dpi=300)
plt.close(fig)

print("✅ Saved plots to logs/images/")  