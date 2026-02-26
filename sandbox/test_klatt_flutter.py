from letalker.pitches.FlutteredPitch import FlutteredPitch

import numpy as np
from matplotlib import pyplot as plt


n = 100000
fo = 100
fs = 44100

t = np.arange(n) / fs
phi = 2 * np.pi * fo * t

flutter = FlutteredPitch(phi0="random")

fo_flutter = flutter.modify_pitch(t, fo)

phi_flutter = flutter.modify_phase(t, phi, fo)

dphi_flutter = np.diff(phi_flutter) * fs / 2 / np.pi


fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(phi_flutter - phi)
axes[0].set_ylabel("phase change")
axes[1].plot(fo_flutter)
axes[1].plot(dphi_flutter, "--")
axes[1].axhline(fo, ls=":", c="C2")
axes[1].set_ylabel("instantaneous frequency")

plt.show()
