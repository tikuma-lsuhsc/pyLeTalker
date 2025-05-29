from matplotlib import pyplot as plt
import numpy as np
from letalker.function_generators import *
from letalker.constants import fs

n = fs  # 1 second demo
t = np.arange(n) / fs

fo_mean = [100,124]
fo_div = 20
fo = FlutterGenerator(fo_div,0.1,16,bias=fo_mean,shape=2)
xi_ref = SineGenerator(fo)
print(xi_ref(n).shape)
exit()


analytic = False
gens = [
    FlutterGenerator(0.1, 0.1, 3, phi0=0),
    FlutterGenerator(0.1, 0.1, 3, phi0=0, shape=2),
    FlutterGenerator(1, [1, 0.4], phi0=0),
    FlutterGenerator(1, [1, 0.4], phi0=0, bias=0.5),
    FlutterGenerator(1, [1, 0.4], phi0=0, transition_time=0.1),
    FlutterGenerator(1, [1, 0.4], phi0=0, bias=LineGenerator([0.2, 0.8], [1, 2])),
    FlutterGenerator(1, [1, 0.4], phi0=0, bias=0.5, transition_time=0.1),
    FlutterGenerator(
        1, [1, 0.4], phi0=0, bias=LineGenerator([0.2, 0.8], [1, 2]), transition_time=0.1
    ),
]

fig, axes = plt.subplots(len(gens), 3, sharex=True, squeeze=False)
for i, (raxes, gen) in enumerate(zip(axes, gens)):
    x = gen(n)
    dx = gen.derivative(n, force_time_axis=True)
    ix = gen.antiderivative(n)

    raxes[0].plot(t, x, label="x")
    raxes[1].plot(t, dx)
    raxes[2].plot(t, ix)

    raxes[0].plot(t, x[0] + np.cumsum(dx.real, axis=0) / fs, ":", label="dx")
    raxes[0].plot(t[:-1], np.diff(ix.real, axis=0) * fs, "--", label="ix")
    raxes[0].legend()

    raxes[1].plot(t[:-1], np.diff(x.real, axis=0) * fs, "--", label="x")
    raxes[2].plot(t, ix[0] + np.cumsum(x.real, axis=0) / fs, ":", label="x")


axes[0, 0].set_title("function")
axes[0, 1].set_title("derivative")
axes[0, 2].set_title("antiderivative")

for ax in axes[-1]:
    ax.set_xlabel("time (s)")

plt.show()
