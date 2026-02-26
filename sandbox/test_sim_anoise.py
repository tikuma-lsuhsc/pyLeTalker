from matplotlib import pyplot as plt

import numpy as np
import numba as nb

import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

# from math import ceil, log2
# from scipy.signal import get_window

import letalker as sv
from letalker.constants import fs, male_vf_params, vocaltract_areas, PL


N = 11000

t = np.arange(N) / fs

vf_params = {
    **male_vf_params,
    # "xi0": None,  # prephonatory displacements of the edge of the vf at the level of the vocal process
    # "xim": 0.1,  # common amplitude of the upper and lower edges of the folds
    # "Qa": 0.3,  # abduction quotient: xi02/xim (Titze 84 Eq (8))
    # "Qs": 3.0,  # shape quotient: (xi01-xi02)/xim (Titze 84 Eq (9))
    # "Qp": 0.2,  # phase quotient: phi/2pi (Titze 84 Eq (10))
    # "Qb": 1.0,  # bulging quotient: (|xi01-xi02|/2-(xib-xi02))/xim
}

fo = 101

psd_level = sv.LeTalkerAspirationNoise.psd_level

# lips, vf, lungs = sv.sim_kinematic(
#     N,
#     fo,
#     **vf_params,
#     vocaltract={"areas": "aa", "atten": 0.001},
#     trachea={"atten": 0.001},
#     aspiration_noise={"noise_source": psd_level},
# )
lips, vf, lungs, vf_hout, lip_nout = sv.sim_kinematic_sep(
    N,
    fo,
    **vf_params,
    vocaltract={"areas": "aa", "atten": 0.001},
    trachea={"atten": 0.001},
    aspiration_noise={"noise_source": psd_level},
)

t = t[1000:]
po_harmonics = lips.pout[1000:]
po_noise = lip_nout[1000:, 0]

nfft = 1024 * 8

plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.plot(t, po_harmonics, label="harmonics")
plt.plot(t, po_noise, label="noise")
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.magnitude_spectrum(po_harmonics, fs, pad_to=nfft, scale="dB")
plt.magnitude_spectrum(po_noise, fs, pad_to=nfft, scale="dB")
plt.grid()
plt.legend()

# if vf.noise_gen:
#     nfft = round(2 ** np.ceil(np.log2(N)))
#     plt.figure()
#     plt.plot(vf.ug * vf.noise_gen.nu_inv / vf.L)
#     plt.axhline(np.sqrt(vf.noise_gen.RE2b))
plt.show()
