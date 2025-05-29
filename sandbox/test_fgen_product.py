from matplotlib import pyplot as plt
import numpy as np
from letalker.function_generators import *
from letalker.constants import fs

n = fs  # 1 second demo
t = np.arange(n) / fs

fo = FlutterGenerator(3, 1, 3, bias=10)
print(fo._flutter_args)
fm = ProductGenerator(0.5, fo)

# gen1 = SineGenerator(fo=fo)
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t, fo(n))
axes[0].plot(t, fm(n)*2)
axes[1].plot(t, fo.antiderivative(n))
axes[1].plot(t, fm.antiderivative(n)*2)
# axes[1].plot(t[:-1], np.diff(gen1.phase(n)) * gen1.fs / 2 / np.pi)


plt.show()
