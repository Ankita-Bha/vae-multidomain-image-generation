class BetaScheduler:
    def __init__(self, start=0.0, end=1.0, n_steps=10000):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1
        beta = self.start + (self.end - self.start) * min(
            self.step_count / self.n_steps, 1.0
        )
        return beta
