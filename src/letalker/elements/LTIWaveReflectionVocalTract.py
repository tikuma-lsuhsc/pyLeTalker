from __future__ import annotations

from numpy.typing import NDArray

from .abc import Lips, Lungs, VocalFolds
from .LeTalkerVocalTract import LeTalkerVocalTract


class LTIWaveReflectionVocalTract(LeTalkerVocalTract):
    """Wave-reflection vocal tract model (Liljencrants, 1985; Story, 1995)"""

    class Runner: ...

    source: VocalFolds | Lungs
    sink: Lips | VocalFolds

    def generate_sim_params(self, n: int, n0: int = 0, **_) -> tuple[NDArray, ...]:

        areas = self.areas(n, n0)

        alpha = 1 - self._atten / areas**0.5
        r = (areas[..., :-1] - areas[..., 1:]) / (areas[..., :-1] + areas[..., 1:])

        return (alpha[..., ::2], alpha[..., 1::2], r[..., ::2], r[..., 1::2])
