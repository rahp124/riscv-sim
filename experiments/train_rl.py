import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from envs.tinyrv_env import TinyRVEnv

def main():
    env = TinyRVEnv(max_steps=50, prog_len=8)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./data/tensorboard/"
    )

    print("[INFO] Training PPO agent...")
    model.learn(total_timesteps=5000)

    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_tinyrv"
    model.save(model_path)
    print(f"[INFO] Model saved at {model_path}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"[RESULT] PPO mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    print("Generated program (indices):", action)

    instrs = [env.instr_set[idx] for idx in action]
    print("Generated program (encodings):", [hex(i) for i in instrs])

    _, reward, _, _ = env.step(action)
    print(f"Coverage achieved: {reward}")
    print("\n[Disassembly]")
    pc = 0
    for i, enc in enumerate(instrs):
        asm = env.decode_instr(enc)
        print(f"{pc:08x}: {asm}  # [{i}]")
        pc += 4

    ppo_rewards = []
    for _ in range(20):
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _ = env.step(action)
        ppo_rewards.append(reward)

    os.makedirs("data", exist_ok=True)
    np.save("data/ppo_rewards.npy", np.array(ppo_rewards))

if __name__ == "__main__":
    main()
