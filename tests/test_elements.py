import pytest
import numpy as np

# import numba as nb

from letalker import (
    LeTalkerLungs,
    OpenLungs,
    LeTalkerVocalTract,
    LossyCylinderVocalTract,
    LeTalkerLips,
    KinematicVocalFolds,
    KlattAspirationNoise,
)
from letalker.core import use_numba, using_numba

from letalker.constants import male_vf_params


# @nb.njit
def run_edge(runner, x):
    y = np.empty_like(x)
    for i, xi in enumerate(x):
        y[i, 0] = runner.step(i, xi[0])
    return y


# @nb.njit
def run(runner, x):
    y = np.empty_like(x)
    for i, xi in enumerate(x):
        y[i, 0], y[i, 1] = runner.step(i, xi[0], xi[1])
    return y


@pytest.mark.parametrize(
    "CLS, args, kwargs",
    [
        (LeTalkerLungs, (), {}),
        (OpenLungs, (), {}),
        (LeTalkerVocalTract, ("aa",), {}),
        (LeTalkerVocalTract, ("aa",), {"log_sections": True}),
        (LeTalkerLips, (), {}),
        (KinematicVocalFolds, (100,), male_vf_params),
        (KinematicVocalFolds, (100,), {**male_vf_params, "aspiration_noise": True}),
        (
            KinematicVocalFolds,
            (100,),
            {**male_vf_params, "aspiration_noise": KlattAspirationNoise()},
        ),
        (LossyCylinderVocalTract, (17.5, 1.7), {}),
    ],
)
def test_runner(CLS, args, kwargs):
    element = CLS(*args, **kwargs)

    N = 10
    runner = element.create_runner(N)

    n_ports = (not element.is_sink) + (not element.is_source)

    x = np.random.randn(N, n_ports)

    if n_ports == 1:
        run_edge(runner, x)
    else:
        run(runner, x)

    results = element.create_result(runner)
    assert isinstance(results, CLS.Results)
