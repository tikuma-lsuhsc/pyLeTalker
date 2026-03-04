import numpy as np
from numpy.typing import NDArray

from letalker import LeTalkerVocalTract


class FastRunner:
    n: int
    alph_odd: NDArray  # odd-section propagation gains
    alph_even: NDArray  # even-section propagation gains
    r_odd: NDArray  # odd-junction reflection coefficients
    r_even: NDArray  # even-junction reflection coefficients
    s: np.ndarray

    def __init__(
        self,
        nb_steps: int,
        s: NDArray,
        alph_odd: NDArray,
        alph_even: NDArray,
        r_odd: NDArray,
        r_even: NDArray,
    ):
        super().__init__()

        self.n = nb_steps
        self.alph_odd = alph_odd
        self.alph_even = alph_even
        self.r_odd = r_odd
        self.r_even = r_even
        self.s = s

    def step(self, i: int, f1: float, bK: float) -> tuple[float, float]:
        """time update

        Parameters
        ----------
        f1
            forward pressure input
        bK
            backward pressure input

        Returns
        -------
            fK
                forward pressure output
            b1
                backward pressure output

        """

        alph_odd = self.alph_odd[i if i < self.alph_odd.shape[0] else -1]
        alph_even = self.alph_even[i if i < self.alph_even.shape[0] else -1]
        r_odd = self.r_odd[i if i < self.r_odd.shape[0] else -1]
        r_even = self.r_even[i if i < self.r_even.shape[0] else -1]
        s = self.s

        ns = r_even.shape[0]
        f_even = s[:ns]
        b_odd = s[ns:]

        f_odd = np.empty_like(alph_odd)
        b_even = np.empty_like(alph_even)

        # ---even junctions [(F2,B3),(F4,B5),...,(FK-2,BK-1)]->[(F3,B2),(F5,B4),...]--- */
        Psi = (f_even - b_odd) * r_even
        f_odd[0] = f1
        f_odd[1:] = f_even + Psi
        b_even[:-1] = b_odd + Psi
        b_even[-1] = bK

        f_odd *= alph_odd
        b_even *= alph_even

        # ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */
        Psi = (f_odd - b_even) * r_odd  # [F3,F5,...] - [B2, B4, ...]
        f_odd += Psi
        b_even += Psi

        # --- attenuate pressure to "pre-"propagate signal to the output of the section
        f_odd *= alph_even
        b_even *= alph_odd

        s[:ns] = f_odd[:-1]
        s[ns:] = b_even[1:]

        return f_odd[-1], b_even[0]


def test_runner():

    n = 100
    vt = LeTalkerVocalTract("aa")
    cpp_runner = vt.create_runner(n)

    params = vt.generate_sim_params(n)
    py_runner = FastRunner(n, np.zeros(vt.nb_states), *params)

    fsg = np.random.rand(n) * 100
    beplx = np.random.rand(n) * 100
    cpp_out = np.empty((n, 2))
    py_out = np.empty((n, 2))
    for i, (f, b) in enumerate(zip(fsg, beplx)):
        cpp_out[i] = py_runner.step(i, f, b)
        py_out[i] = cpp_runner.step(i, f, b)

    assert np.array_equal(cpp_out, py_out)
