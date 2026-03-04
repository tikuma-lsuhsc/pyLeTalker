import numpy as np
from numpy.typing import NDArray

from letalker.elements import LeTalkerLips


class Runner:
    n: int
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    s: np.ndarray
    pout: np.ndarray
    uout: np.ndarray

    def __init__(self, nb_steps: int, s: NDArray, A: NDArray, b: NDArray, c: NDArray):
        super().__init__()

        self.n = nb_steps
        self.A = np.ascontiguousarray(A.transpose(0, 2, 1))
        self.b = b
        self.c = c
        self.s = np.ascontiguousarray(s)
        self.pout = np.empty(nb_steps)
        self.uout = np.empty(nb_steps)

    def step(self, i: int, flip: float, blip: float) -> float:

        Ai = self.A[i if i < self.A.shape[0] else -1]
        bi = self.b[i if i < self.b.shape[0] else -1]
        c = self.c[i if i < self.c.shape[0] else -1]
        s = self.s

        # states: Po_prev, b_prev, f_prev
        s[:2] = s @ Ai + bi * flip
        s[-1] = flip

        self.pout[i] = s[0]
        self.uout[i] = c * (flip - s[1])

        return 0.0, s[1]


def test_runner():

    n = 100
    lips = LeTalkerLips(2.4)
    cpp_runner = lips.create_runner(n)

    params = lips.generate_sim_params(n)
    py_runner = Runner(n, np.zeros(3), *params)

    flip = np.random.rand(n) * 100
    blip = np.empty((n, 2))
    for i, f in enumerate(flip):
        _, blip[i, 1] = py_runner.step(i, f, 0.0)
        _, blip[i, 0] = cpp_runner.step(i, f, 0.0)

    lips.create_result(cpp_runner)

    assert np.array_equal(blip[:, 0], blip[:, 1])
