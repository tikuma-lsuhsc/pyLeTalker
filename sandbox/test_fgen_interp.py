from matplotlib import pyplot as plt
import numpy as np
from letalker.function_generators import *
from letalker.constants import fs

n = fs  # 1 second demo
t = np.arange(n) / fs

tp = [0.1, 0.35, 0.64, 0.87]
xp = [[0, 4], [1, 2], [4, 1], [3, 3]]
gen0 = Interpolator(tp, xp, degree=3)

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

    axes[0, nu - 1].plot(t, x0 + np.cumsum(dx, 0) / fs)
    axes[1, nu - 1].plot(t[:-1], np.diff(ix, axis=0) * fs)

    x0 = dx[0]

from scipy.interpolate import make_interp_spline

x = np.linspace(0, 2 * np.pi, 10)
y = np.array([np.sin(x), np.cos(x)])

plt.figure()
ax = plt.axes(projection="3d")
xx = np.linspace(0, 20 * np.pi, 1000)
bspl = make_interp_spline(x, y, k=5, bc_type="periodic", axis=1)
ax.plot3D(xx, *bspl(xx))
ax.scatter3D(x, *y, color="red")


plt.show()
