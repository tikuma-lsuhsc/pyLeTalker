from itertools import product
from matplotlib import pyplot as plt
import numpy as np

from letalker.function_generators import *
from letalker.constants import fs

n = fs // 10  # 1 second demo
t = np.arange(n) / fs

gen = RosenbergGenerator(100)

f = gen(n)

plt.plot(t, f)
plt.show()
