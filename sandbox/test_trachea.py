import numpy as np
import numba as nb
from os import path
from matplotlib import pyplot as plt
from oct2py import octave

import os

rootdir = ".." if os.getcwd().endswith("tests") else "."

from letalker import LeTalkerVocalTract, LeTalkerLungs
from letalker.constants import fs, vocaltract_areas

N = 1000
t = np.arange(N) / fs
x = np.random.randn(N)

lungs = LeTalkerLungs()
trachea = LeTalkerVocalTract("trach")


@nb.njit
def sim(N, lung_runner, trachea_runner):

    flung = blung = 0.0
    y = np.empty(N)
    for i in range(N):
        flung = lung_runner.step(i, blung)
        y[i], blung = trachea_runner.step(i, flung, 0.0)

    return y, lung_runner, trachea_runner


y, *runners = sim(N, lungs.create_runner(N), trachea.create_runner(N))

lung_res = lungs.create_result(runners[0])

plt.plot(lung_res.plung, label="lungs")
plt.plot(y, label="subglottal")
plt.legend()


# from oct2py import octave

# octave.addpath(os.path.join(rootdir, "LeTalker1.22"))
# octave.eval("warning ('off', 'Octave:data-file-in-path')")
# ymat = octave.test_trachea(tr_areas, lungs.PL, nout=1)

# plt.plot(ymat, "--.", label="subglottal (MATLAB)")
plt.show()
