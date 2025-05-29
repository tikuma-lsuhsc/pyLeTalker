import numpy as np
from matplotlib import pyplot as plt

from letalker import ConstantPitch, LinearPitch, SinusoidalPitch
from letalker.constants import fs

N = fs
fo = 100

pitches = [
    ConstantPitch(fo),
    ConstantPitch(fo, flutter=True),
    LinearPitch(fo, 1),
    SinusoidalPitch(fo, 1, 0.5),
]

t = pitches[0].ts(N)
fs = pitches[0].fs

plt.subplots(2, 1, sharex=True)
plt.subplot(2, 1, 1)
plt.plot(t, np.transpose([pitch.eval_pitch(N, expand=True) for pitch in pitches]))
plt.subplot(2, 1, 2)
plt.plot(
    t[:-1],
    np.diff([pitch.eval_phase(N) for pitch in pitches], axis=-1).T / 2 / np.pi * fs,
)
plt.show()
