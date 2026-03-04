import numpy as np
from letalker._backend import NoFlowNoiseRunner
from numpy.typing import NDArray

from letalker.elements import VocalFoldsAg


class Runner:
    n: int
    R: np.ndarray
    Qa: np.ndarray
    a: np.ndarray
    rhoca_sg: np.ndarray
    rhoca_eplx: np.ndarray
    ug: np.ndarray
    psg: np.ndarray
    peplx: np.ndarray

    def __init__(
        self,
        nb_steps: int,
        s_in: NDArray,
        anoise,
        R: NDArray,
        Qa: NDArray,
        a: NDArray,
        rhoca_sg: NDArray,
        rhoca_eplx: NDArray,
    ):
        super().__init__()

        self.n = nb_steps
        self.R = R
        self.Qa = Qa
        self.a = a
        self.rhoca_sg = rhoca_sg
        self.rhoca_eplx = rhoca_eplx
        self.ug = np.empty(nb_steps)
        self.psg = np.empty(nb_steps)
        self.peplx = np.empty(nb_steps)
        self._anoise = anoise

    def step(self, i: int, fsg: float, beplx: float) -> tuple[float, float]:

        R = self.R[i if i < self.R.shape[0] else -1]
        Qa = self.Qa[i if i < self.Qa.shape[0] else -1]
        a = self.a[i if i < self.a.shape[0] else -1]
        rhoca_eplx = self.rhoca_eplx[i if i < self.rhoca_eplx.shape[0] else -1]
        rhoca_sg = self.rhoca_sg[i if i < self.rhoca_sg.shape[0] else -1]

        # compute the glottal flow
        Q = Qa * (fsg - beplx)
        if fsg < beplx:
            # depends on whether ug is positive or negative
            a = -a
            Q = -Q
        ug = a * ((R**2 + Q) ** 0.5 - R)

        ug += self._anoise.step(i, ug, [])

        self.ug[i] = ug

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
    vf = VocalFoldsAg(ug)
    cpp_runner = vf.create_runner(n, noise_free=True)

    params = vf.generate_sim_params(n)
    py_runner = Runner(n, None, NoFlowNoiseRunner(), *params)

    fsg = np.random.rand(n) * 100
    beplx = np.random.rand(n) * 100
    for i, (f, b) in enumerate(zip(fsg, beplx)):
        py_runner.step(i, f, b)
        cpp_runner.step(i, f, b)

    vf.create_result(cpp_runner)

    assert np.array_equal(py_runner.peplx, cpp_runner.peplx)
    assert np.array_equal(py_runner.psg, cpp_runner.psg)
