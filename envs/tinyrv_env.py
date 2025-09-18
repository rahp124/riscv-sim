import gym
import numpy as np
from gym import spaces
from sim.tinyrv_wrapper import TinyRVWrapper


class TinyRVEnv(gym.Env):
    def __init__(self, max_steps=50, prog_len=8):
        super().__init__()
        self.wrapper = TinyRVWrapper(max_steps=max_steps)
        self.prog_len = prog_len

        self.instr_set = [
            0x00000013,  # nop (addi x0, x0, 0)
            0x00100093,  # addi x1, x0, 1
            0x00208113,  # addi x2, x1, 2
            0x403101b3,  # sub x3, x2, x3
            0x00012203,  # lw x4, 0(x2)
            0x00412423,  # sw x4, 4(x2)
            0xfe211ae3,  # beq x2, x1, -4
            0x0000006f,  # jal x0, 0
        ]

        self.action_space = spaces.MultiDiscrete([len(self.instr_set)] * prog_len)

        self.observation_space = spaces.Dict({
            "pc": spaces.Discrete(2**16),
            "regs": spaces.Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(32,),
                dtype=np.int32,
            ),
        })

        self.decoder = {
            0x00000013: "nop        (addi x0, x0, 0)",
            0x00100093: "addi x1, x0, 1",
            0x00208113: "addi x2, x1, 2",
            0x403101b3: "sub  x3, x2, x3",
            0x00012203: "lw   x4, 0(x2)",
            0x00412423: "sw   x4, 4(x2)",
            0xfe211ae3: "beq  x2, x1, -4",
            0x0000006f: "jal  x0, 0",
        }

    def reset(self):
        state = self.wrapper.reset()
        return {"pc": state["pc"], "regs": np.array(state["regs"], dtype=np.int32)}

    def step(self, action):
        instrs = [self.instr_set[idx] for idx in action]
        program_bytes = b"".join([instr.to_bytes(4, "little") for instr in instrs])

        state = self.wrapper.reset(program_bytes=program_bytes)

        done = False
        while not done:
            state, done = self.wrapper.step()

        obs = {"pc": state["pc"], "regs": np.array(state["regs"], dtype=np.int32)}
        reward = len(state["coverage"])
        return obs, reward, True, {}

    def decode_instr(self, encoding: int) -> str:
        """Return human-readable assembly string for an instruction encoding."""
        return self.decoder.get(encoding, f"UNKNOWN(0x{encoding:x})")
