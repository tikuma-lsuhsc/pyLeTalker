from matplotlib import pyplot as plt
import numpy as np
from letalker.function_generators import *
from letalker.constants import fs

n = fs  # 1 second demo
t = np.arange(n) / fs

ttypes = [
    "step",
    "linear",
    "raised_cos",
    "exp_decay",
    "exp_decay_rev",
    "logistic",
    "atan",
    "tanh",
    "erf",
]

gens = [
    StepGenerator(
        [1 / 3, 2 / 3],
        [[0, 1.0], [1.0, 0.3], [0.5, 0.5]],
        transition_type=ttype,
        transition_time_constant=0.1,
    )
    for ttype in ttypes
]

fig, axes = plt.subplots(len(gens), 3, sharex=True, squeeze=False)
for raxes, ttype, gen in zip(axes, ttypes, gens):
    x = gen(n)
    dx = gen.derivative(n, force_time_axis=True)
    ix = gen.antiderivative(n)
    raxes[0].plot(t, x)
    raxes[1].plot(t, dx)
    raxes[2].plot(t, ix)

    raxes[0].plot(t, x[0, :] + np.cumsum(dx.real, axis=0) / fs, ":", label="dx")
    raxes[0].plot(t[:-1], np.diff(ix.real, axis=0) * fs, "--", label="ix")
    raxes[0].legend()

    raxes[1].plot(t[:-1], np.diff(x.real, axis=0) * fs, "--", label="x")
    raxes[2].plot(t, ix[0, :] + np.cumsum(x.real, axis=0) / fs, ":", label="x")

axes[0, 0].set_title("function")
axes[0, 1].set_title("derivative")
axes[0, 2].set_title("antiderivative")

for raxes, ttype in zip(axes, ttypes):
    raxes[0].set_ylabel(ttype)

for ax in axes[-1]:
    ax.set_xlabel("time (s)")

plt.show()
