from matplotlib import pyplot as plt

import numpy as np
import numba as nb

import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

# from math import ceil, log2
# from scipy.signal import get_window

from letalker.elements import (
    KinematicVocalFolds,
    LeTalkerVocalTract,
    LeTalkerLips,
    LeTalkerLungs,
    LeTalkerAspirationNoise,
)
from letalker.function_generators import (
    SineGenerator,
    ModulatedSineGenerator,
    StepGenerator,
)
from letalker.constants import fs, male_vf_params, vocaltract_areas, PL
from letalker.sim import sim, sim_kinematic


def test_sim():
    # from simvoice import sim

    vt_areas = vocaltract_areas["aa"]
    tr_areas = vocaltract_areas["trach"]

    N = 4000  # fs  # 10000  # fs

    t = np.arange(N) / fs

    fo = 101

    m_freq = 1 / 2
    extent = 0.2

    sine = SineGenerator(fo)  # ,phi0 = -pi/2)
    am_sine = ModulatedSineGenerator(fo, m_freq, am_extent=extent)
    fm_sine = ModulatedSineGenerator(fo, m_freq, fm_extent=extent)
    amfm_sine = ModulatedSineGenerator(fo, m_freq, am_extent=extent, fm_extent=extent)

    lung_pressure = StepGenerator(PL)

    vocaltract = LeTalkerVocalTract(vt_areas)
    trachea = LeTalkerVocalTract(tr_areas)

    male_vf_params["L0"] = 1
    vf = KinematicVocalFolds(
        sine(t, analytic=True),
        sine.fo.eval_pitch(t),
        **male_vf_params,
    )
    print(vf.L)

    nfsrc = LeTalkerAspirationNoise.default_noise_source
    nf = LeTalkerAspirationNoise(100 * nfsrc(t))

    vf.add_aspiration_noise(nf)

    lips = LeTalkerLips(vt_areas[-1])
    lungs = LeTalkerLungs(lung_pressure(t))

    po, parts = sim(N, vf, vocaltract, trachea, lungs, lips)

    lips, vf, lungs = (parts[c] for c in ("lips", "vocalfolds", "lungs"))
    ug, psg, pe = vf.ug, vf.psg, vf.peplx

    plt.subplots(4, 1, sharex=True)
    plt.subplot(4, 1, 1)
    plt.plot(t, vf.glottal_area.value, label="area")
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t, ug, label="glottal flow")
    plt.plot(t, vf.noise_source.unoise, label="aspiration noise component")
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(t, lung_pressure(t).value, t, psg, t, pe)
    plt.legend(["PL", "Psg", "Pe"])
    plt.grid()
    plt.subplot(4, 1, 4)
    plt.plot(t, po, label="acoustic")
    plt.grid()
    plt.legend()

    nfft = round(2 ** np.ceil(np.log2(N)))
    plt.figure()
    # plt.magnitude_spectrum(vf.ug, fs, pad_to=nfft, scale="dB")
    plt.plot(vf.ug * vf.noise_gen.nu_inv / vf.L)
    plt.axhline(np.sqrt(vf.noise_gen.RE2b))
    plt.show()

    # plt.figure()
    # plt.plot(t, x._val, t, y._val, t, z._val, t, w._val)
    # plt.figure()
    # nfft = 2 ** ceil(log2(N * 4))
    # win = get_window("hamming", N)
    # plt.magnitude_spectrum(x._val[:, 0], fs, pad_to=nfft, scale="dB", window=win)
    # plt.magnitude_spectrum(y._val[:, 0], fs, pad_to=nfft, scale="dB", window=win)
    # plt.magnitude_spectrum(z._val[:, 0], fs, pad_to=nfft, scale="dB", window=win)
    # plt.magnitude_spectrum(w._val[:, 0], fs, pad_to=nfft, scale="dB", window=win)
    # plt.xlim([0, 1000])
    # plt.show()


def test_sim_kinematic():

    N = 4000

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

    po, parts = sim_kinematic(
        N, 101, **vf_params, vocaltract="aa", aspiration_noise=True
    )

    vf, lungs = (parts[c] for c in ("vocalfolds", "lungs"))
    ug, psg, pe = vf.ug, vf.psg, vf.peplx

    plt.subplots(4, 1, sharex=True)
    plt.subplot(4, 1, 1)
    plt.plot(t, vf.glottal_area, label="area")
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t, ug, label="glottal flow")
    if vf.aspiration_noise:
        plt.plot(t, vf.aspiration_noise.unoise, label="aspiration noise component")
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(t, lungs.plung, t, psg, t, pe)
    plt.legend(["PL", "Psg", "Pe"])
    plt.grid()
    plt.subplot(4, 1, 4)
    plt.plot(t, po, label="acoustic")
    plt.grid()
    plt.legend()

    if vf.aspiration_noise:
        nfft = round(2 ** np.ceil(np.log2(N)))
        plt.figure()
        plt.plot(vf.aspiration_noise.re2)
        plt.axhline(np.sqrt(vf.aspiration_noise.re2b))
    plt.show()


if __name__ == "__main__":
    test_sim_kinematic()
