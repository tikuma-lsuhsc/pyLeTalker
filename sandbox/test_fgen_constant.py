from matplotlib import pyplot as plt
import numpy as np
from letalker.function_generators import *
from letalker.constants import fs

n = fs  # 1 second demo
t = np.arange(n) / fs

# test taper mixin
gen0 = Constant([0.98, 0.5], transition_time=0.5)
print(gen0(n).shape)
print(gen0.derivative(n).shape)
print(gen0.antiderivative(n).shape)

gen0 = Constant(0.98)

fig, axes = plt.subplots(2, 4, sharex=True)
x = gen0(n, force_time_axis=True)
x0 = x[0]
axes[0, 0].plot(t, x)
axes[1, 0].plot(t, x)
for nu in range(1, 4):
    dx = gen0.derivative(n, nu=nu, force_time_axis=True)
    ix = gen0.antiderivative(n, nu=nu)
    axes[0, nu].plot(t, dx)
    axes[1, nu].plot(t, ix)

    axes[0, nu - 1].plot(t, np.cumsum(dx) / fs)
    axes[1, nu - 1].plot(t[:-1], np.diff(ix) / np.diff(t))

    x0 = ix[0]


n = fs  # 1 second demo
t = np.arange(n) / fs

cgen0 = Constant(1)
cgen1 = Constant(1, transition_time=0.25)
cgen2 = Constant(1, transition_time=0.5)
cgen3 = Constant(1, transition_time=0.75)

fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
axes[0].plot(t, cgen0(n, force_time_axis=True), label="transition_time=None")
axes[0].legend()
axes[1].plot(t, cgen1(n), label="transition_time=0.25")
axes[1].legend()
axes[2].plot(t, cgen2(n), label="transition_time=0.5")
axes[2].legend()
axes[3].plot(t, cgen3(n), label="transition_time=0.75")
axes[3].legend()
axes[3].set_xlabel("time (s)")

plt.show()
