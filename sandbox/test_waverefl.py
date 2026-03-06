import numpy as np
from numpy.typing import NDArray

from letalker import LeTalkerVocalTract


class Runner:
    n: int
    alph_odd: NDArray  # odd-section propagation gains
    alph_even: NDArray  # even-section propagation gains
    r_odd: NDArray  # odd-junction reflection coefficients
    r_even: NDArray  # even-junction reflection coefficients
    s: np.ndarray
    p_sections: np.ndarray  # internal pressure (columns: forward/backward)

    def __init__(
        self,
        nb_steps: int,
        s: NDArray,
        alph_odd: NDArray,
        alph_even: NDArray,
        r_odd: NDArray,
        r_even: NDArray,
        log_sections: bool,
    ):
        super().__init__()

        self.n = nb_steps
        self.alph_odd = alph_odd
        self.alph_even = alph_even
        self.r_odd = r_odd
        self.r_even = r_even
        self.s = s
        self.log = log_sections

        n_sections = r_odd.shape[-1] + r_even.shape[-1]
        self.p_sections = np.empty((nb_steps, 2, n_sections))

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

        if self.log:
            self.p_sections[i, 0, 1::2] = f_even
            self.p_sections[i, 1, 1::2] = b_even[:-1]

        f_odd *= alph_odd  # f_even input to odd junction
        b_even *= alph_even  # b_even input to odd junction

        # ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */
        Psi = (f_odd - b_even) * r_odd  # [F3,F5,...] - [B2, B4, ...]
        f_odd += Psi
        b_even += Psi

        # grab odd-stage output forward pressure
        if self.log:
            self.p_sections[i, 0, ::2] = f_odd

        # --- attenuate pressure to "pre-"propagate signal to the output of the section
        f_odd *= alph_even  # misnamed, f_even next state
        b_even *= alph_odd  # misnamed, b_odd next state

        if self.log:
            self.p_sections[i, 1, ::2] = b_even  # odd-stage output backward pressure

        s[:ns] = f_odd[:-1]
        s[ns:] = b_even[1:]

        return f_odd[-1], b_even[0]


n = 1000
vt = LeTalkerVocalTract("aa", log_sections=True)
cpp_runner = vt.create_runner(n)

params = vt.generate_sim_params(n)
py_runner = Runner(n, np.zeros(vt.nb_states), *params)

fsg = np.random.rand(n) * 100
beplx = np.random.rand(n) * 100

fout = np.empty((n, 2))
bout = np.empty((n, 2))
for i, (f, b) in enumerate(zip(fsg, beplx)):
    fout[i, 0], bout[i, 0] = py_runner.step(i, f, b)
    fout[i, 1], bout[i, 1] = cpp_runner.step(i, f, b)

vt.create_result(cpp_runner)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 3)
axes[0].plot(fout, ".-")
axes[1].plot(bout, ".-")


axes[2].plot(cpp_runner.p_sections[:, 1, :2], ".-")
axes[2].plot(py_runner.p_sections[:, 1, :2], "x--")


plt.show()


# letalker_dir = path.join(
#     "." if getcwd().endswith("tests") else "tests", "matlab", "LeTalker1.22"
# )

# print(letalker_dir)

# octave.addpath(letalker_dir)
# octave.eval("warning ('off', 'Octave:data-file-in-path')")
# ymat = octave.test_waverefl(areas, x, nout=1)

# plt.plot(ymat, "--.")
# plt.show()
