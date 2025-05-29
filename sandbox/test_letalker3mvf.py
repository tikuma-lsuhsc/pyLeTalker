from matplotlib import pyplot as plt
import numpy as np
from letalker import LeTalkerVocalFolds, use_numba
from letalker.elements.LeTalkerVocalFolds import calc_stress

use_numba(False)

N = 10

vf = LeTalkerVocalFolds()

runner = vf.create_runner(N)
K = runner.K[0]
B = runner.B[0]
w = runner.w[0]

f = np.zeros((N, 3))
# f[:, :2] = np.random.rand(N, 2)*0.1
f[:] = 0
y0 = np.random.rand(6)*0.05
y0[-1] += max(y0[:2])

y = np.empty((N + 1, 6))
y[0, :] = y0
for i, fi in enumerate(f):
    y[i + 1] = runner.rk3m(y[i], i, fi, K, B, w)

plt.plot(y[:, :3])
plt.show()


def test_calc_pressures():
    # tangent calculation in calc_pressure is not consistent with letalker
    runner = vf.create_runner(N)

    x = np.random.rand(9)

    outvars = ["a1", "a2", "f1", "f2"]

    out = runner.calc_pressures(*x)

    from oct2py import octave

    octave.addpath("pyLeTalker/sandbox/matlab/LeTalker1.22")
    octave.eval("warning ('off', 'Octave:data-file-in-path');")
    p = octave.test_calc_pressures(*x)

    for i, var in enumerate(outvars):
        print(f"{p[var]=} vs {out[i]}")


def test_rules_consconv():

    L, T, zn, m, k, kc, b = vf.rules_consconv(1)
    # print(m, k, kc, b)

    from oct2py import octave

    octave.addpath("pyLeTalker/sandbox/matlab/LeTalker1.22")
    octave.eval("warning ('off', 'Octave:data-file-in-path');")
    p = octave.test_rules_consconv()

    print(
        L - p["L"],
        T - p["T"],
        zn - p["zn"],
        m - [p["m1"], p["m2"], p["M"]],
        k - [p["k1"], p["k2"], p["K"]],
        kc - p["kc"],
        b - [p["b1"], p["b2"], p["B"]],
    )

    print(f"{b=}")
    print(f'{p["b1"]=}, {p["b2"]=}, {p["B"]=}')


def test_calc_stress():
    eps = np.random.rand()
    sigmuc, sigl, sigp = calc_stress(eps)
    print(f"{sigmuc=}, {sigl=}, {sigp=}")

    # L, T, zn, m, k, kc, b = vf.rules_consconv(1)
    # print(m, k, kc, b)

    from oct2py import octave

    octave.addpath("pyLeTalker/sandbox/matlab/LeTalker1.22")
    octave.eval("warning ('off', 'Octave:data-file-in-path');")
    # p = octave.test_rules_consconv()
    p = octave.test_calc_stress(eps)
    # p['L']==p.L


def test_eom():
    runner = vf.create_runner(N)
    yo = np.random.randn(6)
    f = np.random.randn(3)
    tt = 0
    dyo = runner.eom_3m(yo, tt, f, runner.K[0], runner.B[0], runner.w[0])
    print(dyo)

    L, T, zn, m, k, kc, b = vf.rules_consconv()
    print(m, k, kc, b)

    from oct2py import octave

    octave.addpath("pyLeTalker/sandbox/matlab/LeTalker1.22")
    octave.eval("warning ('off', 'Octave:data-file-in-path')")
    print(octave.test_eom(yo, dyo, tt, m, k, kc, b, f)[0] - dyo)
    # ymat, ugmat = octave.test_eom(
    #     x[:, 0], x[:, 1], vf.glottal_area, vf.epiglottal_area, vf.subglottal_area, nout=2
    # )
