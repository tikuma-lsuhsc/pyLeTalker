"""Default constants

This module defines most of the default values used by various pyLeTalker classes
and functions.

"""

# References
# ----------

# [1] I. R. Titze, “Physiologic and acoustic differences between male and female voices,”
#     J. Acoust. Soc. Am., vol. 85, no. 4, pp. 1699–1707, Apr. 1989, doi: 10.1121/1.397959.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args
from numpy.typing import NDArray

from functools import lru_cache
from math import pi

_2pi = 2 * pi

c: float = 35000  #: speed of sound in cm/s
rho_air: float = 0.00114  #: air density in g/cm^3 (= kg/mm^3)
rho_tis: float = 1.04  #: tissue density in g/cm^3 (= kg/mm^3)
mu: float = 0.000186  #: air viscosity in dyne-s/cm  (= Pa·s = kg/(m·s))
nu: float = rho_air / mu  #: kinematic viscosity (m²/s)
fs: int = 44100  #: sampling rate in S/s - must be fixed

male_vf_params = {"L0": 1.42, "T0": 0.3, "fo2L": (4, 9.4), "xim": 0.12}  #: kinematic vocal fold model parameters of a male
female_vf_params = {"L0": 0.86, "T0": 0.2, "fo2L": (5.5, 7.6), "xim": 0.1}  #: kinematic vocal fold model parameters of a female

# fo2L: (A, B), A - kPa, B - unitless

PL: float = 7840  #: respiratory driving pressure from lungs (dyn/cm²)

kinematic_params = {
    "Qa": 0.3,  #: abduction quotient: xi02/xim (Titze 84 Eq (8))
    "Qs": 2.0,  #: shape quotient: (xi01-xi02)/xim (Titze 84 Eq (9))
    "Qb": 1.0,  #: bulging quotient: (|xi01-xi02|/2-(xib-xi02))/xim
    "Qp": 0.2,  #: phase quotient: phi/2pi (Titze 84 Eq (10))
    "Qnp": 0.7,  #: nodal point ratio
}

#### ASPIRATION NOISE DEFAULT PARAMETERS

noise_REc: float = 1200.0  #: aspiration noise Reynolds number threshold
noise_psd_level: float = (4e-6) ** 2 / 12  #: aspiration noise PSD peak level
noise_bpass = (
    2,
    (300, 3000),
    "bandpass",
)  #: aspiration noise coloring bandpass filter arguments
noise_dist = "uniform"  #: aspiration noise innovation process distribution

#### VOCAL TRACT DEFAULT PARAMETERS

vt_atten: float = 0.002  #: vocal tract attenuation factor


@lru_cache(1)
def _load_vocaltract_data():
    from scipy.io.matlab import loadmat
    from os import path
    import numpy as np

    datadir = path.join(path.split(__file__)[0], "data")

    raw = loadmat(path.join(datadir, "bs_origvowels.mat")) | loadmat(
        path.join(datadir, "LeTalkerTrachea.mat")
    )
    return {
        k: v.reshape(-1)
        for k, v in raw.items()
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and k != "areas"
    }


TwoLetterVowelLiteral = Literal[tuple(_load_vocaltract_data().keys())]  # Two-letter vowels

if TYPE_CHECKING:
    vocaltract_areas: dict[TwoLetterVowelLiteral, NDArray]

vocaltract_resolution = c / (2 * fs)  #: vocal tract tube segment length


def __getattr__(name):
    if name == "vocaltract_areas":
        d = _load_vocaltract_data()
        return d
    if name == "vocaltract_names":
        l = get_args(TwoLetterVowelLiteral)
        return l
    raise AttributeError()
