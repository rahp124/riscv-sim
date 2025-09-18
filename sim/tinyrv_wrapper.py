import tinyrv

class TinyRVWrapper:
    def __init__(self, xlen=32, max_steps=1000):
        self.xlen = xlen
        self.max_steps = max_steps
        self.reset()

    def reset(self, program_bytes=None):
        self.rv = tinyrv.sim(xlen=self.xlen)
        self.pc_history = set()
        self.steps = 0
        if program_bytes is not None:
            self.rv.copy_in(0, program_bytes)
        return self._snapshot()

    def step(self):
        if self.steps >= self.max_steps:
            return self._snapshot(), True
        self.rv.step()
        self.steps += 1
        return self._snapshot(), False

    def _snapshot(self):
        pc = self.rv.pc
        regs = tuple(self.rv.x)
        self.pc_history.add(pc)
        return {
            "pc": pc,
            "regs": regs,
            "coverage": set(self.pc_history),
            "step": self.steps
        }
