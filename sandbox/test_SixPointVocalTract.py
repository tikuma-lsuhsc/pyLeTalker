from typing import get_args
import numpy as np
from matplotlib import pyplot as plt
# from oct2py import octave

from letalker.elements import SixPointVocalTract

# from letalker.constants import vocaltract_areas

# N = 1000
# x = np.random.randn(N, 2)
# y = np.empty_like(x)
# vocaltract = SixPointVocalTract.from_preset('a:')
# vt_runner = vocaltract.create_runner(N)

# for i, (feplx, blip) in enumerate(x):
#     y[i, 0], y[i, 1] = vt_runner.step(i, feplx, blip)


for v in get_args(SixPointVocalTract.VocalTractSound)[14:15]:
    for data_type in ('geom','optim'):
        for reduced in (False,True):
            vocaltract = SixPointVocalTract.from_preset(v,data_type, reduced)
            plt.step(vocaltract.z, vocaltract.areas()[0],label=f'{v}:{data_type}:{reduced}')

plt.legend()
plt.show()
