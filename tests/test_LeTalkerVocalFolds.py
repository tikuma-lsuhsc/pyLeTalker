import pytest

import letalker as lt
import numpy as np


def test_LeTalkerThreeMassVocalFolds():

    act = None  #: float | NDArray | SampleGenerator | None = None,
    ata = None  #: float | NDArray | SampleGenerator | None = None,
    alc = None  #: float | NDArray | SampleGenerator | None = None,
    x0 = None  #: NDArray | SampleGenerator | None = None,
    Lo = None  #: float | None = None,
    To = None  #: float | None = None,
    Dmo = None  #: float | None = None,  # resting depth of muscle
    Dlo = None  #: float | None = None,  # resting depth of deep layer
    Dco = None  #: float | None = None,  # resting depth of the cover
    zeta = None  #: NDArray | SampleGenerator | None = None

    vf = lt.LeTalkerVocalFolds(
        act=act,
        ata=ata,
        alc=alc,
        x0=x0,
        Lo=Lo,
        To=To,
        Dmo=Dmo,
        Dlo=Dlo,
        Dco=Dco,
        zeta=zeta,
    )

    Fin = 1.0
    Bin = 0.1

    runner = vf.create_runner(1)
    out = runner.step(0, Fin, Bin)
    res = vf.create_result(runner)
