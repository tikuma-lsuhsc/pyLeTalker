import numpy as np
from letalker._backend import PyFlowNoiseRunnerBase
from numpy.typing import NDArray

from letalker.elements import LeTalkerAspirationNoise


class Runner(PyFlowNoiseRunnerBase):
    n: int
    nuL_inv: NDArray
    nf: NDArray
    re2b: float
    ug_noise: NDArray

    def __init__(
        self,
        nb_steps: int,
        s: NDArray,
        nuL_inv: NDArray,
        nf: NDArray,
        RE2b: float,
    ):

        super().__init__()

        self.n = nb_steps
        self.nuL_inv = nuL_inv
        self.nf = nf
        self.re2b = RE2b
        self.re2 = np.empty(nb_steps)
        self.ug_noise = np.empty(nb_steps)

    def step(self, i: int, ug: float) -> float:

        nuL_inv = self.nuL_inv[i if i < self.nuL_inv.shape[0] else -1]
        nf = self.nf[i if i < self.nf.shape[0] else -1]
        RE2b = self.re2b
        self.re2[i] = RE2 = (ug * nuL_inv) ** 2.0

        self.ug_noise[i] = u = ((RE2 - RE2b) * nf) if RE2 > RE2b else 0.0

        return u


def test_runner():

    n = 100
    noise = LeTalkerAspirationNoise()
    cpp_runner = noise.create_runner(n, length=2.5)
    py_runner = Runner(n, None, cpp_runner.nuL_inv, cpp_runner.nf, cpp_runner.re2b)

    ug = np.random.rand(n) * 2000
    for i, u in enumerate(ug):
        py_runner.step(i, u)
        cpp_runner.step(i, u, [])

    noise.create_result(cpp_runner)

    assert np.array_equal(py_runner.re2, cpp_runner.re2)
    assert np.array_equal(py_runner.ug_noise, cpp_runner.ug_noise)
