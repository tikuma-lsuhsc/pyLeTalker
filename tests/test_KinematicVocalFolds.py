import pytest

from letalker.elements.KinematicVocalFolds import *

# from letalker import ClampedInterpolator
from letalker.constants import male_vf_params
from letalker import SineGenerator, ClampedInterpolator

# xi0: ArrayLike | FunctionGenerator,
# Qb: ArrayLike | FunctionGenerator | None = None,
# z: NDArray
# nb_samples: int | None
# n0: int = 0


@pytest.mark.parametrize(
    "xi0, Qb, n, res_ndim",
    [
        ([1.0, 0.8], None, None, 1),
        ([1.0, 0.8], 0.6, None, 1),
        ([1.0, 0.6, 0.8], None, None, 1),
        ([1.0, 0.6, 0.8], 0.4, None, TypeError),
        (ClampedInterpolator([0, 1], [[1.0, 0.8], [0.9, 0.85]]), None, 10, 2),
        (ClampedInterpolator([0, 1], [[1.0, 0.8], [0.9, 0.85]]), 0.6, 10, 2),
        (
            ClampedInterpolator([0, 1], [[1.0, 0.8], [0.9, 0.85]]),
            ClampedInterpolator([0, 1], [0.6, 0.65]),
            10,
            2,
        ),
        ([1.0, 0.8], ClampedInterpolator([0, 1], [0.6, 0.65]), 10, 2),
    ],
)
def test_xi0ZfunPoly(xi0, Qb, n, res_ndim):

    z = np.linspace(0, 1, 10)

    if isinstance(res_ndim, type) and issubclass(res_ndim, Exception):
        with pytest.raises(res_ndim):
            xi0_zfun = Xi0ZFunPoly(xi0, Qb)
            xi0_zfun(z, n)
    else:
        xi0_zfun = Xi0ZFunPoly(xi0, Qb)
        assert xi0_zfun(z, n).ndim == res_ndim


@pytest.mark.parametrize(
    "xim, Qa, Qs, n, res_ndim",
    [
        (0.1, None, None, None, 1),
        (0.1, 1.0, 0.5, None, 1),
        (0.1, 1.0, 0.5, 10, 1),
        (0.1, [1.0, 0.8], ClampedInterpolator([0, 1], [0.6, 0.65]), 10, 3),
        (0.1, ClampedInterpolator([0, 1], [0.6, 0.65]), [1.0, 0.8], 10, 3),
        (
            0.1,
            ClampedInterpolator([0, 1], [0.6, 0.65]),
            ClampedInterpolator([0, 1], [0.6, 0.65]),
            10,
            2,
        ),
    ],
)
def test_xi0ZfunPolyRel(xim, Qa, Qs, n, res_ndim):

    z = np.linspace(0, 1, 10)

    if isinstance(res_ndim, type) and issubclass(res_ndim, Exception):
        with pytest.raises(res_ndim):
            xi0_zfun = Xi0ZFunPolyRel(xim, Qa, Qs)
            xi0_zfun(z, n)
    else:
        xi0_zfun = Xi0ZFunPolyRel(xim, Qa, Qs)
        assert xi0_zfun(z, n).ndim == res_ndim


def test_KinematicVocalFolds():
    fo = 101
    sine = SineGenerator(fo)
    N = 100
    vf = KinematicVocalFolds(sine, **male_vf_params)
    vf.generate_sim_params(N)
    ga = vf.glottal_area(N)
    ca = vf.contact_area(N)

    vf = KinematicVocalFolds(sine, **male_vf_params, aspiration_noise=True)
    assert vf.noise_model is not None
