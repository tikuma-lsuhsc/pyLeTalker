from __future__ import annotations

from numpy.typing import ArrayLike

from ..constants import TwoLetterVowelLiteral, vocaltract_areas
from ..constants import fs as default_fs
from ..function_generators.abc import SampleGenerator
from .WaveReflectionVocalTract import WaveReflectionVocalTract


class LeTalkerVocalTract(WaveReflectionVocalTract):
    """Wave-reflection vocal tract model (Liljencrants, 1985; Story, 1995)"""

    def __init__(
        self,
        areas: ArrayLike | TwoLetterVowelLiteral | SampleGenerator,
        atten: float | None = None,
        log_sections: bool = False,
        min_areas: float | None = None,
    ):
        """Wave-reflection vocal tract model

        :param areas: cross-sectional areas of vocal tract segments
        :param atten: per-section attenuation factor, defaults to None
        :param log_sections: `True` to log the incidental pressure at every vocal tract segment, defaults to False
        :param min_areas: minimum allowable cross-sectional area to enforce positive area, defaults to None
        """
        # """Wave-reflection vocal tract model

        # The wave-reflection vocal tract model represents the vocal tract as a
        # series of short lossy tube segments. Each segment has a fixed length
        # :math:`c/fs/2` (by default, 0385 cm).

        # Parameters
        # ----------
        # areas
        #     cross-sectional areas of vocal tract segments.
        # atten, optional
        #     per-section attenuation factor, by default `None`
        # log_sections, optional
        #     `True` to log the incidental pressure at every vocal tract segment,
        #     by default `False`
        # min_areas, optional
        #     minimum allowable cross-sectional area to enforce positive area,
        #     by default `None` to use `1e-6`.

        # """
        if isinstance(areas, str):
            areas = vocaltract_areas[areas]
            if self.fs != default_fs:
                raise ValueError(
                    f"Incorrect {self.fs=} to use default vocal tract area."
                )

        super().__init__(areas, atten, log_sections, min_areas)
