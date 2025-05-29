import numpy as np
from os import path, getcwd
from matplotlib import pyplot as plt
from oct2py import octave

from letalker.elements import LeTalkerVocalTract

from letalker.constants import vocaltract_areas

areas = vocaltract_areas["aa"]

N = 1000
x = np.random.randn(N, 2)

vocaltract = LeTalkerVocalTract(areas, log_sections=True)

y = np.empty_like(x)

vt_runner = vocaltract.create_runner(N)

for i, (feplx, blip) in enumerate(x):
    y[i, 0], y[i, 1] = vt_runner.step(i, feplx, blip)

plt.plot(y)
plt.show()


# letalker_dir = path.join(
#     "." if getcwd().endswith("tests") else "tests", "matlab", "LeTalker1.22"
# )

# print(letalker_dir)

# octave.addpath(letalker_dir)
# octave.eval("warning ('off', 'Octave:data-file-in-path')")
# ymat = octave.test_waverefl(areas, x, nout=1)

# plt.plot(ymat, "--.")
# plt.show()
