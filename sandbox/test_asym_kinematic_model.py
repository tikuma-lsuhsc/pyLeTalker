from matplotlib import pyplot as plt
import numpy as np
from math import ceil, log2
from scipy.signal import get_window

# from oct2py import octave
# from os import path, getcwd

from letalker import (
    AsymmetricKinematicVocalFolds,
    SineGenerator,
    ModulatedSineGenerator,
)
from letalker.constants import fs, male_vf_params, rho_air, c, vocaltract_areas, PL

N = 4000

t = np.arange(N) / fs

fo = 50.5

# m_freq = 1 / 2
# extent = 0.2

# sine = SineGenerator(fo)
# am_sine = ModulatedSineGenerator(fo, m_freq, am_extent=extent)
# fm_sine = ModulatedSineGenerator(fo, m_freq, fm_extent=extent)
# amfm_sine = ModulatedSineGenerator(fo, m_freq, am_extent=extent, fm_extent=extent)

# x = sine(t)
# y = am_sine(t)
# z = fm_sine(t)
# w = amfm_sine(t)

Ae = vocaltract_areas["aa"][0]
As = vocaltract_areas["trach"][-1]

vf = AsymmetricKinematicVocalFolds([fo * 2, fo * 3], As=As, Ae=Ae, **male_vf_params)

print(vf.area_epiglottal, vf.area_subglottal)
print(Ae, As)

plt.subplots(2, 1, sharex=False)
plt.subplot(2, 1, 1)
plt.plot(vf.glottal_area(N), label="glottal area")
plt.subplot(2, 1, 2)
plt.magnitude_spectrum(vf.glottal_area(N), fs, scale="dB", pad_to=44100)
plt.xlim(0, 1000)
# plt.plot(vf.contact_area(N), label="contact area")
plt.show()
exit()

N = 1000
x = PL * 0.1 * np.random.randn(N, 2)  # random inputs
# x = np.zeros((N, 2))  # random inputs
x[:, 0] += PL
y = np.empty_like(x)

vf_runner = vf.create_runner(N)
for i, xi in enumerate(x):
    y[i, 0], y[i, 1] = vf_runner.step(i, *xi)
vf_res = vf.create_result(vf_runner)

plt.figure(2)
plt.plot(vf_res.ug)
plt.figure(3)
plt.plot(y)

# letalker_dir = path.join(
#     "." if getcwd().endswith("tests") else "tests", "matlab", "LeTalker1.22"
# )

# print(letalker_dir)

# octave.addpath(letalker_dir)
# octave.eval("warning ('off', 'Octave:data-file-in-path')")
# ymat, ugmat = octave.test_vfflow(
#     x[:, 0], x[:, 1], vf.glottal_area, vf.epiglottal_area, vf.subglottal_area, nout=2
# )

# plt.figure(2)
# plt.plot(ugmat, "--.")
# plt.figure(3)
# plt.plot(ymat, "--.")
# plt.legend(['feplx (py)','bsg (py)','feplx (ml)','bsg (ml)'])
plt.show()
