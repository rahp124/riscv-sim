## RISC-V Instruction Sequence Generation with Reinforcement Learning

## UCSC ACM

---

Overview

This project explores how RL can be applied to generate instruction sequences for a RISC-V CPU simulator(TinyRV). It illustrates how RL can guide program generation to systematically explore CPU behaviors, highlighting the potential for AI assisted verification.

We compare random instruction generation against a PPO trained RL agent, demonstrating how AI can improve code coverage in a toy CPU.

## TinyRV Environment

TinyRVEnv is a custom Gymnasium environment wrapping TinyRVWrapper.

Action space: A discrete set of RISC-V instructions.

Observation space: CPU registers and program counter.

Reward: Number of unique instructions executed (“coverage”).

Episode: Each program runs for a fixed number of instructions or until completion.

The environment also supports pretty disassembly, translating instruction encodings into human readable assembly.

## Training the RL Agent

Use train_rl.py to train a PPO agent:

The script trains the agent for 5000 timesteps on programs of length 8.

After training, it evaluates the agent on 20 episodes.

Outputs include:

Generated program indices and RISC-V encodings

Coverage achieved

Pretty disassembly

Example output:

[RESULT] PPO mean reward: 51.00 ± 0.00
Generated program (indices): [5 5 0 0 1 6 0 3]
Generated program (encodings): ['0x412423', '0x412423', '0x13', ...]
Coverage achieved: 51

[Disassembly]
00000000: sw x4, 4(x2) # [0]
00000004: sw x4, 4(x2) # [1]
00000008: nop (addi x0, x0, 0) # [2]
...

## Random Baseline

Use random_baseline.py to generate programs randomly:

Produces random instruction sequences of the same length.

Measures coverage over 100 episodes.

Saves results to data/random_rewards.npy.

Example output:

[RANDOM] Mean coverage: 13.08 ± 20.81

## Comparing PPO vs Random

plot_results.py creates visual comparisons:

Histogram: Distribution of coverage for random vs PPO programs.

Training curve: PPO mean reward over training iterations.

Results
Method Mean Coverage Std Dev
Random 13.08 20.81
PPO 51.00 0.00

## Takeaways

---

RL significantly outperforms random generation: PPO achieves nearly four times the average coverage of random instruction sequences.

Consistency: PPO produces stable coverage, while random sequences are highly variable.

By learning patterns, PPO avoids redundant or ineffective instructions, generating sequences that exercise more of the CPU.
