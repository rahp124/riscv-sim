import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.tinyrv_env import TinyRVEnv

env = TinyRVEnv(max_steps=50, prog_len=8)
rewards = []

for _ in range(100):
    obs, _ = env.reset()  
    action = env.action_space.sample()
    _, reward, done, _ = env.step(action)
    rewards.append(reward)

rewards = np.array(rewards)
print(f"[RANDOM] Mean coverage: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

# Save rewards for plotting
os.makedirs("data", exist_ok=True)
np.save("data/random_rewards.npy", rewards)
print("[RANDOM] Saved rewards to data/random_rewards.npy")
