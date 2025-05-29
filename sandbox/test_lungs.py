from matplotlib import pyplot as plt
import numpy as np

from letalker import LeTalkerLungs, OpenLungs

N = 1000

ltlung = LeTalkerLungs()
oplung = OpenLungs()

b2 = np.random.randn(N) * 0.5 * ltlung.lung_pressure.levels[0]
f1 = np.empty((N, 2))
res = np.empty((N, 0))

lt_runner = ltlung.create_runner(N)
op_runner = oplung.create_runner(N)

for n, b2n in enumerate(b2):
    f1[n, 0] = lt_runner.step(n, b2n)
    f1[n, 1] = op_runner.step(n, b2n)

lt_res = ltlung.create_result(lt_runner)
op_res = oplung.create_result(op_runner)

plt.plot(lt_res.plung, ".-")
plt.plot(op_res.plung, ".-")
plt.plot(ltlung.lung_pressure(N), ".-")
# plt.plot(f1, ".-")
plt.show()
