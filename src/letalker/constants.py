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
import numpy as np

_2pi = 2 * pi

c: float = 35000  #: speed of sound in cm/s
rho_air: float = 0.00114  #: air density in g/cm^3 (= kg/mm^3)
rho_tis: float = 1.04  #: tissue density in g/cm^3 (= kg/mm^3)
mu: float = 0.000186  #: air viscosity in dyne-s/cm  (= Pa·s = kg/(m·s))
nu: float = rho_air / mu  #: kinematic viscosity (m²/s)
fs: int = 44100  #: sampling rate in S/s - must be fixed

male_vf_params = {
    "L0": 1.42,
    "T0": 0.3,
    "fo2L": (4, 9.4),
    "xim": 0.12,
}  #: kinematic vocal fold model parameters of a male
female_vf_params = {
    "L0": 0.86,
    "T0": 0.2,
    "fo2L": (5.5, 7.6),
    "xim": 0.1,
}  #: kinematic vocal fold model parameters of a female

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

    datadir = path.join(path.split(__file__)[0], "data")

    raw = loadmat(path.join(datadir, "bs_origvowels.mat")) | loadmat(
        path.join(datadir, "LeTalkerTrachea.mat")
    )
    return {
        k: v.reshape(-1)
        for k, v in raw.items()
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and k != "areas"
    }


TwoLetterVowelLiteral = Literal[
    tuple(_load_vocaltract_data().keys())
]  # Two-letter vowels

if TYPE_CHECKING:
    vocaltract_areas: dict[TwoLetterVowelLiteral, NDArray]

vocaltract_resolution = c / (2 * fs)  #: vocal tract tube segment length

########################

# The Six-Parameter Model data provided by Stone, Marxen, & Birkholz 2018

# fmt: off
SMB2018VocalTractSound = Literal['a:','e:','i:','o:','u:','E:','oe:','y:','a','E','I','O','U','oE','Y','@','f','l','s','S','C','x']
#                        Literal['aː','eː','iː','oː','uː','ɛː','ø:', 'y:','a','ɛ','ɪ','ɔ','ʊ','œ', 'ʏ','ə','f','l','s','ʃ','ç','x']
#                        Literal['AAː','EYː','IYː','OWː','UWː','EHː','ø:', 'y:','AA','EH','ɪ','ɔ','ʊ','œ', 'ʏ','AX','f','l','s','ʃ','ç','x']
# fmt: on

smb_xlar_default = 2.0
smb_Alar_default = 1.5125
smb_nlarp_default = 1.0
smb_xp_default = 3.0948
smb_xa_coefs = np.array([-2.347, -0.061, -2.052, -0.159, 1.161, 0.143])

@lru_cache(4)
def smb_vt_area_data(data_type: Literal["geom", "optim"], reduced: bool) -> NDArray:
    """Stone-Marxen-Birkholz six-point vocal tract model preset parameters

    Args:
        data_type: "geom" for the geometrically fitted configurations or
                   "optim" for the perceptually optimized configurations
        reduced: True to return the reduced 11-parameter configurations or
                 False to return the full 16-parameter configurations

    Returns:
        A numpy record array, containing 22 fields, one for each sound
        specified in `SMB2018VocalTractSound`. Each field consists of either 11
        or 16 parameters, depending on `reduced` argument:

        | i | name |
        |---|------|
        | 0 | Ap   |
        | 1 | npc  |
        | 2 | xc   |
        | 3 | Ac   |
        | 4 | nca  |
        | 5 | Aa   |
        | 6 | nain |
        | 7 | xin  |
        | 8 | Ain  |
        | 9 | xlip |
        |10 | Alip |
        |11 | xlar |
        |12 | Alar |
        |13 | nlarp|
        |14 | xp   |
        |15 | xa   |

    """
    from os import path
    import numpy as np

    csvfile = path.join(
        path.split(__file__)[0],
        "data",
        f"smb2018_{data_type}_{'reduced' if reduced else 'full'}.csv",
    )

    data = np.rec.fromarrays(
        np.loadtxt(csvfile, delimiter=","), names=get_args(SMB2018VocalTractSound)
    )

    if not reduced:
        data = np.concatenate([data[4:9], data[10:], data[:4], data[[9]]], axis=0)

    return data


def __getattr__(name):
    if name == "vocaltract_areas":
        d = _load_vocaltract_data()
        return d
    if name == "vocaltract_names":
        l = get_args(TwoLetterVowelLiteral)
        return l
    raise AttributeError()
