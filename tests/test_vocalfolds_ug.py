import numpy as np
from letalker._backend import NoFlowNoiseRunner
from numpy.typing import NDArray

from letalker.elements import VocalFoldsUg


class Runner:
    n: int
    _ug: np.ndarray
    rhoca_sg: np.ndarray
    rhoca_eplx: np.ndarray
    psg: np.ndarray
    peplx: np.ndarray

    def __init__(
        self,
        nb_steps: int,
        s_in: NDArray,
        anoise,
        ug: NDArray,
        rhoca_sg: NDArray,
        rhoca_eplx: NDArray,
    ):
        super().__init__()

        self.n = nb_steps
        self._ug = ug
        self.rhoca_sg = rhoca_sg
        self.rhoca_eplx = rhoca_eplx
        self.psg = np.empty(nb_steps)
        self.peplx = np.empty(nb_steps)
        self._anoise = anoise

    def step(self, i: int, fsg: float, beplx: float) -> tuple[float, float]:

        ug = self._ug[i if i < self._ug.shape[0] else -1]
        rhoca_eplx = self.rhoca_eplx[i if i < self.rhoca_eplx.shape[0] else -1]
        rhoca_sg = self.rhoca_sg[i if i < self.rhoca_sg.shape[0] else -1]

        ug += self._anoise.step(i, ug, [])

        # compute the forward pressure in epilarnx
        feplx = beplx + ug * rhoca_eplx

        # compute the backward pressure in subglottis
        bsg = fsg - ug * rhoca_sg

        self.psg[i] = fsg + bsg
        self.peplx[i] = feplx + beplx

        return feplx, bsg


def test_runner():

    n = 100
    ug = np.random.rand(n) * 2000
    vf = VocalFoldsUg(ug)
    cpp_runner = vf.create_runner(n, noise_free=True)

    params = vf.generate_sim_params(n)
    py_runner = Runner(n, None, NoFlowNoiseRunner(), *params)

    fsg = np.random.rand(n) * 100
    beplx = np.random.rand(n) * 100
    for i, (f, b) in enumerate(zip(fsg, beplx)):
        py_runner.step(i, f, b)
        cpp_runner.step(i, f, b)

    assert np.array_equal(py_runner.peplx, cpp_runner.peplx)
    assert np.array_equal(py_runner.psg, cpp_runner.psg)
