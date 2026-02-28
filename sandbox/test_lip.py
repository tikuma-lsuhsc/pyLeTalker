import numpy as np
from matplotlib import pyplot as plt

from letalker.constants import fs, vocaltract_areas
from letalker.elements import LeTalkerLips

areas = vocaltract_areas["aa"]

N = 1000
x = np.random.randn(N)

lips = LeTalkerLips(areas[-1])

y = np.empty_like(x)

lip_runner = lips.create_runner(N)


for i, flip in enumerate(x):
    y[i] = lip_runner.step(i, flip)
lip_res = lips.create_result(lip_runner)


import scipy.signal as sps

b, a = next(lips.iter_tf())
f, H = sps.freqz(b[0], a, fs=fs, worN=1024 * 8)
f, H1 = sps.freqz(b[1], a, fs=fs, worN=1024 * 8)

plt.plot(f, np.log10(np.abs(H)), f, np.log10(np.abs(H1)))

plt.figure()
plt.plot(lip_res.pout)
plt.plot(lip_res.uout)
plt.plot(y)
plt.show()

if False:
    from os import getcwd, path

    from oct2py import octave

    letalker_dir = path.join(
        "." if getcwd().endswith("tests") else "tests", "matlab", "LeTalker1.22"
    )

    print(letalker_dir)

    octave.addpath(letalker_dir)
    octave.eval("warning ('off', 'Octave:data-file-in-path')")
    ymat = octave.test_lip(areas[-1], x, nout=1)

    plt.plot(ymat, "--.")
    plt.show()
