import pytest

from itertools import product
import numpy as np

# from oct2py import octave
# from os import path, getcwd

from letalker.function_generators.abc import SampleGenerator
from letalker.elements.KinematicVocalFolds import (
    Xi0ZFunPoly,
    Xi0ZFunPolyRel,
    _load_asym_param,
)
import letalker as lt

n = 10
nz = 15
z = np.linspace(0, 1, nz, True)


@pytest.mark.parametrize(
    "gen,ndim,force_data_axis,shape",
    [
        (lt.Constant(1), 0, False, (1,)),
        (lt.Constant([1]), 0, False, (1, 1)),
        (lt.Constant([1, 1]), 0, False, (2, 1)),
        (lt.SineGenerator(100), 0, False, (n,)),
        (lt.SineGenerator([100]), 0, False, (n, 1)),
        (lt.SineGenerator([100, 100]), 0, False, (2, n)),
        (lt.Constant(1), 1, False, (1,)),
        (lt.Constant([1]), 1, False, (1, 1)),
        (lt.Constant([1, 1]), 1, False, (1, 2)),
        (lt.SineGenerator(100), 1, False, (n,)),
        (lt.SineGenerator([100]), 1, False, (n, 1)),
        (lt.SineGenerator([100, 100]), 1, False, (n, 2)),
        (lt.Constant(1), 0, True, (1,)),
        (lt.Constant([1]), 0, True, (1, 1)),
        (lt.Constant([1, 1]), 0, True, (2, 1)),
        (lt.SineGenerator(100), 0, True, (n,)),
        (lt.SineGenerator([100]), 0, True, (n, 1)),
        (lt.SineGenerator([100, 100]), 0, True, (2, n)),
        (lt.Constant(1), 1, True, (1, 1)),
        (lt.Constant([1]), 1, True, (1, 1)),
        (lt.Constant([1, 1]), 1, True, (1, 2)),
        (lt.SineGenerator(100), 1, True, (n, 1)),
        (lt.SineGenerator([100]), 1, True, (n, 1)),
        (lt.SineGenerator([100, 100]), 1, True, (n, 2)),
    ],
)
def test_load_asym_param(gen, ndim, force_data_axis, shape):

    assert (
        _load_asym_param(gen, ndim, n, force_data_axis=force_data_axis).shape == shape
    )


@pytest.mark.parametrize(
    "xi0",
    [
        lt.Constant([0.4, 0.1, 0.1]),  # fixed symmetric
        lt.Constant([[0.4, 0.1, 0.1], [0.3, 0.15, 0.1]]),  # fixed asymmetric
        lt.LineGenerator([0, 1], [[0.4, 0.1, 0.1], [0.6, 0.2, 0.15]]),  # dynamic sym
        lt.LineGenerator(  # dynamic asym
            [0, 1],
            [[[0.4, 0.1, 0.1], [0.3, 0.15, 0.1]], [[0.4, 0.1, 0.1], [0.3, 0.15, 0.1]]],
        ),
    ],
)
def test_Xi0ZFunPoly_noQb(xi0):
    xi0z = Xi0ZFunPoly(xi0)
    xi0 = xi0z(z, n)
    print(xi0.shape)


xi0s = [
    lt.Constant([0.4, 0.1]),  # fixed symmetric
    lt.Constant([[0.4, 0.1], [0.3, 0.1]]),  # fixed asymmetric
    lt.LineGenerator([0, 1], [[0.4, 0.1], [0.6, 0.15]]),  # dynamic sym
    lt.LineGenerator(  # dynamic asym
        [0, 1],
        [[[0.4, 0.1], [0.3, 0.1]], [[0.4, 0.1], [0.3, 0.1]]],
    ),
]

Qbs = [
    None,
    lt.Constant(1),
    lt.Constant([1]),
    lt.Constant([0.4, 0.1]),
    lt.LineGenerator([0, 1], [0, 1]),
    lt.LineGenerator([0, 1], [[0.4, 0.1], [0.3, 0.1]]),
]


@pytest.mark.parametrize("xi0,Qb", product(xi0s, Qbs))
def test_Xi0ZFunPoly(xi0, Qb):

    is_dynamic = not all(p is None or p.is_fixed for p in (xi0, Qb))
    is_lateral = (xi0.ndim > 1 and xi0.shape[0] > 1) or (
        Qb is not None and Qb.ndim and Qb.shape[0] > 1
    )

    out_shape = (n if is_dynamic else 1, nz)
    if is_lateral:
        out_shape = (2, *out_shape)
    elif not is_dynamic:    
        out_shape = out_shape[1:]

    xi0z = Xi0ZFunPoly(xi0, Qb)
    xi0 = xi0z(z, n)
    assert xi0.shape == out_shape


######################


@pytest.mark.parametrize("xim, Qa, Qs, Qb", product(Qbs[1:], Qbs, Qbs, Qbs))
def test_Xi0ZFunPolyRel(xim, Qa, Qs, Qb):

    xi0z = Xi0ZFunPolyRel(xim, Qa, Qs, Qb)

    is_dynamic = not all(p is None or p.is_fixed for p in (xim, Qa, Qs, Qb))
    is_lateral = any(
        p is not None and p.ndim and p.shape[0] > 1 for p in (xim, Qa, Qs, Qb)
    )

    out_shape = (nz,)
    if is_dynamic:
        out_shape = (n, *out_shape)
    if is_lateral:
        out_shape = (2, *out_shape) if is_dynamic else (2, 1, *out_shape)

    xi0 = xi0z(z, n)
    assert xi0.shape == out_shape


if __name__ == "__main__":
    print("testing _load_asym_param")

    gens = [
        lt.Constant(1),  # 0 - scalar, fixed, sym
        lt.Constant([1]),  # 1 - fixed, sym
        lt.Constant([1, 1]),  # 2 - fixed, asym
        lt.LineGenerator([0, 1], [0, 1]),  # 3 - dynamic,sym
        lt.LineGenerator([0, 1], [[0.4, 0.1], [0.3, 0.1]]),  # 4 - dynamic, asym
    ]

    print(_load_asym_param(gens[0], 0, 4).shape)
    print(_load_asym_param(gens[1], 0, 4).shape)
    print(_load_asym_param(gens[2], 0, 4).shape)
    print(_load_asym_param(gens[2], 0, 4, force_time_axis=False).shape)
    print(_load_asym_param(gens[3], 0, 4).shape)
    print(_load_asym_param(gens[3], 0, 4, force_time_axis=False).shape)
    print(_load_asym_param(gens[4], 0, 4).shape)
    print(_load_asym_param(gens[4], 0, 4, force_time_axis=False).shape)
