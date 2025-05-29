import numpy as np
from matplotlib import pyplot as plt

from letalker import LeTalkerVocalTract, LeTalkerLungs
from letalker.core import use_numba

use_numba(True)

lungs = LeTalkerLungs()
vocaltract = LeTalkerVocalTract("aa", log_sections=True)

N = 1000
y = np.empty((N, 2))
flung = blung = fsg = 0.0

vt_runner = vocaltract.create_runner(N)
lung_runner = lungs.create_runner(N)

for i in range(N):
    flung = lung_runner.step(i, blung)
    y[i, 0], y[i, 1] = vt_runner.step(i, flung, 0.0)

vt_res = vocaltract.create_result(vt_runner)
lung_res = lungs.create_result(lung_runner)

plt.plot(lung_res.plung)
plt.plot(y)

plt.subplots(3, 1, sharex=True)
plt.subplot(3, 1, 1)
plt.plot(vt_res.pout_sections)
plt.subplot(3, 1, 2)
plt.plot(vt_res.uout_sections)
plt.subplot(3, 1, 3)
plt.plot(vt_res.re_sections)

plt.show()
