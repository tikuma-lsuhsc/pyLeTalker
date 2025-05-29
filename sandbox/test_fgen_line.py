from matplotlib import pyplot as plt
import numpy as np
from letalker.function_generators import *
from letalker.constants import fs

n = fs  # 1 second demo
t = np.arange(n) / fs

tp = (0.2, 0.8)
xp = (0.98, 0.24)

# test taper mixin
gen0 = LineGenerator(tp, [[0.98, 0.48], [0.24, 0.64]], transition_time=0.5)
print(gen0(n).shape)
print(gen0.derivative(n).shape)
print(gen0.antiderivative(n).shape)


gen0 = LineGenerator(tp, xp)

fig, axes = plt.subplots(2, 4, sharex=True)
x = gen0(n)
x0 = x[0]
axes[0, 0].plot(t, x)
axes[1, 0].plot(t, x)
for nu in range(1, 4):
    dx = gen0.derivative(n, nu=nu)
    ix = gen0.antiderivative(n, nu=nu)
    axes[0, nu].plot(t, dx)
    axes[1, nu].plot(t, ix)

    axes[0, nu - 1].plot(t, x0 + np.cumsum(dx) / fs)
    axes[1, nu - 1].plot(t[:-1], np.diff(ix) / np.diff(t))

    x0 = dx[0]

plt.show()
