from matplotlib import pyplot as plt
import numpy as np

from letalker.function_generators import StepGenerator
from letalker.constants import fs

N = round(fs)

t = np.arange(N) / fs

fo = 101

m_freq = 1 / 2
extent = 0.2


print(t)
stepfun = StepGenerator(1, 0.1)
plt.plot(t, stepfun(N))

stepfun1 = StepGenerator([1, 0, 1], [0.1, 0.3, 0.6])
plt.plot(t, stepfun1(N))
plt.show()
