import pytest
import numpy as np

from letalker.function_generators import WhiteNoiseGenerator, ColoredNoiseGenerator


def test_white_noise():
    ngen = WhiteNoiseGenerator()
    x, zo = ngen.generate(100)
    assert x.shape == (100,)
    x = ngen(100)
    assert x.shape == (100,)
    assert np.array_equal(x, ngen(100))


def test_colored_noise():
    from letalker.constants import noise_bpass

    ngen = ColoredNoiseGenerator(*noise_bpass)
    x, zo = ngen.generate(100)
    ngen.generate(100, zo)
    assert x.shape == (100,)
    x = ngen(100)
    assert x.shape == (100,)
    assert np.array_equal(x, ngen(100))

    x = ngen(100, 120)
    y = ngen(220)
    assert np.array_equal(x, y[120:])
