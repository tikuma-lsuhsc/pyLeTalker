from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from letalker.function_generators import *
from letalker.constants import fs

analytic = False

n = fs  # 1 second demo
t = np.arange(n) / fs


analytic = False

# fos = [10, LineGenerator([0, 1], [0, 10])]
# amps = [1.2, LineGenerator([0.1, 0.9], [0.9, 1.2])]
# offsets = [0.5, LineGenerator([0.2, 0.8], [0, 1.0])]


# for fo, A, bias in product(fos, amps, offsets):
#     gens = [
#         SineGenerator(fo),
#         SineGenerator(fo, A),
#         SineGenerator(fo, bias=bias),
#         SineGenerator(fo, A, bias=bias),
#         SineGenerator(fo, transition_time=0.1),
#         SineGenerator(fo, A, transition_time=0.1),
#         SineGenerator(fo, bias=bias, transition_time=0.1),
#         SineGenerator(fo, A, bias=bias, transition_time=0.1),
#     ]

#     fig, axes = plt.subplots(len(gens), 3, sharex=True, squeeze=False)
#     fig.suptitle(f"fo={fo} | A={A} | bias={bias}")
#     for raxes, gen in zip(axes, gens):
#         x = gen(n)
#         dx = gen.derivative(n, force_time_axis=True)
#         ix = gen.antiderivative(n)

#         raxes[0].plot(t, x, label="x")
#         raxes[1].plot(t, dx)
#         raxes[2].plot(t, ix)

#         raxes[0].plot(t, x[0] + np.cumsum(dx.real, axis=0) / fs, ":", label="dx")
#         raxes[0].plot(t[:-1], np.diff(ix.real, axis=0) * fs, "--", label="ix")
#         raxes[0].legend()

#         raxes[1].plot(t[:-1], np.diff(x.real, axis=0) * fs, "--", label="x")
#         raxes[2].plot(t, ix[0] + np.cumsum(x.real, axis=0) / fs, ":", label="x")

#     for i, raxes in enumerate(axes):
#         raxes[0].set_ylabel(f"#{i}")

#     axes[0, 0].set_title("function")
#     axes[0, 1].set_title("derivative")
#     axes[0, 2].set_title("antiderivative")

#     for ax in axes[-1]:
#         ax.set_xlabel("time (s)")

# test higher order derivatives & anti-derivatives
# gen0 = SineGenerator(fo=[10, 14], A=[1.2, 1.4], phi0=np.pi / 2)
gen0 = SineGenerator(fo=[10, 14], A=[1.2, 1.4], phi0="random")

nu_max = 5
fig, axes = plt.subplots(2, nu_max, sharex=True)
x = gen0(n)
x0 = x[0]
axes[0, 0].plot(t, x)
axes[1, 0].plot(t, x)
for nu in range(1, nu_max):
    dx = gen0.derivative(n, nu=nu, analytic=analytic)
    ix = gen0.antiderivative(n, nu=nu, analytic=analytic)
    axes[0, nu].plot(t, dx.real)
    axes[1, nu].plot(t, ix.real)

    axes[0, nu - 1].plot(t, x0 + np.cumsum(dx.real, axis=0) / fs)
    axes[1, nu - 1].plot(t[:-1], np.diff(ix.real, axis=0) * fs)

    x0 = dx[0]

plt.show()
