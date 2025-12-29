import numpy as np
import random


class Discrete:
    """
    Minimal stand-in for gymnasium.spaces.Discrete used in this project.
    Provides:
      - n: number of discrete actions
      - sample(): uniform random action in [0, n)
    """

    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return random.randrange(self.n)

    @property
    def shape(self):
        # Match gym's attribute; not used heavily here but cheap to provide.
        return ()


class Box:
    """
    Minimal stand-in for gymnasium.spaces.Box used in this project.
    Provides:
      - low, high: bounds (arrays)
      - shape: observation shape
      - dtype: numpy dtype
      - sample(): uniform random sample in [low, high]
    """

    def __init__(self, low, high, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = dtype

        # Allow scalar or array-like bounds.
        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)

    def sample(self):
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


