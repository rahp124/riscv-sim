import numpy as np
import matplotlib.pyplot as plt

ppo_rewards = np.load("data/ppo_rewards.npy")
random_rewards = np.load("data/random_rewards.npy")

plt.figure(figsize=(8,5))
plt.hist(random_rewards, bins=20, alpha=0.6, label="Random", density=True)
plt.hist(ppo_rewards, bins=20, alpha=0.6, label="PPO", density=True)
plt.xlabel("Coverage")
plt.ylabel("Density")
plt.title("Coverage Distribution: PPO vs Random")
plt.legend()
plt.show()

ppo_curve = np.load("data/ppo_training_curve.npy")
plt.figure(figsize=(8,5))
plt.plot(ppo_curve)
plt.xlabel("Training Iteration")
plt.ylabel("Mean Reward")
plt.title("PPO Training Curve")
plt.show()
