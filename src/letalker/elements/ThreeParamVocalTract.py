from __future__ import annotations

from typing_extensions import Self, Literal
from numbers import Number

import numpy as np

from .WaveReflectionVocalTract import WaveReflectionVocalTract
from ._util import smb_vt_area_fun
from ..constants import smb_vt_area_data, SMB2018VocalTractSound


class HornTongueVocalTract(WaveReflectionVocalTract):

    def __init__(
        self,
        Amin: float,
        Xmin: float,
        lhorn: float | tuple[float, float] | None = None,
        Amax: float | tuple[float, float] | None = None,
        l: float | None = None,
        A1: float | None = None,
        l1: float | None = None,
        Alar: float | None = None,
        llar: float | None = None,
        model: (
            Literal["fant", "stevens", "atal", "ishizaka", "lin", "lin2"] | None
        ) = None,
        atten: float | None = None,
        log_sections: bool = False,
        min_areas: float | None = None,
    ):
        """Three-parameter vocal tract models with horn-shaped tongue section

        :param Amin: main tongue body constriction
        :param Xmin: its axial location
        :param lhorn: tongue constriction width (one-sided), defaults to None
        :param Amax: maximum tube area, defaults to None
        :param l: total length, defaults to None
        :param A1: area between lips, defaults to None
        :param l1: lip thickness, defaults to None
        :param Alar: laryngeal pipe area, defaults to None
        :param llar: laryngeal pipe length, defaults to None
        :param atten: per-section attenuation factor, defaults to None
        :param log_sections: `True` to log the incidental pressure at every vocal tract segment, defaults to False
        :param min_areas: minimum allowable cross-sectional area to enforce positive area, defaults to None

        Reference
        ---------

        [1] K. N. Stevens and A. S. House, “Development of a quantitative description
            of vowel articulation,” J. Acoust. Soc. Am., vol. 27, no. 3, pp. 484–493,
            May 1955, doi: 10.1121/1.1907943.
        [2] Q. Lin, “Speech production theory and articulatory speech synthesis,”
            PhD Thesis, Royal Inst. of Technology (KTH), Stockholm, Sweden, 1990.
        [3] G. Fant, The Acoustic Theory of Speech Production. The Hague, the
            Netherlands: Mouton, 1960.
        [4] J. L. Flanagan, K. Ishizaka, and K. L. Shipley, “Signal models for
            low bit-rate coding of speech,” J. Acoust. Soc. Am., vol. 68, no. 3,
            pp. 780–791, Sept. 1980, doi: 10.1121/1.384817.
        [5] B. S. Atal, J. J. Chang, M. V. Mathews, and J. W. Tukey, “Inversion
            of articulatory-to-acoustic transformation in the vocal tract by a
            computer-sorting technique,” J. Acoust. Soc. Am., vol. 63, no. 5,
            pp. 1535–1555, May 1978, doi: 10.1121/1.381848.

        """

        dx = HornTongueVocalTract.dz

        lhorn = np.array([4.75]) if lhorn is None else np.atleast_1d(lhorn)
        Amax = np.array([8.0]) if Amax is None else np.atleast_1d(Amax)

        if l is None:
            l = 17.5
        if A1 is None:
            A1 = 1.0
        if l1 is None:
            l1 = 4.0
        if Alar is None:
            Alar = 0.5
        if llar is None:
            llar = 2.5

        k = round(l / dx / 2) * 2  # number of segments (must be even)
        x = np.arange(k) * dx + dx / 2

        xb = Xmin - lhorn[0]
        xf = Xmin + lhorn[-1]

        A = np.zeros_like(x)

        tf0 = x < llar
        A[tf0] = Alar

        tf1 = x < xb
        tf = ~tf0 & tf1
        A[tf] = Amax[0]

        tf0 = x < xf
        tf = ~tf1 & tf0

        h = np.acosh((Amax / Amin) ** 0.5) * lhorn
        A[tf] = Amin * np.cosh((x[tf] - Xmin) / h) ** 2

        tf1 = x < l - l1
        tf = ~tf0 & tf1
        A[tf] = Amax[-1]
        A[~tf1] = A1

        super().__init__(A, atten, log_sections, min_areas)

    VocalTractSound = SMB2018VocalTractSound

    @staticmethod
    def from_preset(
        sound: SMB2018VocalTractSound,
        data_type: Literal["geom", "optim"] = "optim",
        reduced: bool = True,
        atten: float | None = None,
        log_sections: bool = False,
        min_areas: float | None = None,
    ) -> ThreeParamVocalTract:

        return ThreeParamVocalTract(
            *smb_vt_area_data(data_type, reduced)[sound],
            atten=atten,
            log_sections=log_sections,
            min_areas=min_areas,
        )
