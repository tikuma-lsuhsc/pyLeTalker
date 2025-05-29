"""`~letalker.core` Module

The :py:mod:`~core` module contains functions and classes which controls the
basic flow of the voice synthesis.

.. autosummary::
   :nosignatures:

   get_sampling_rate
   set_sampling_rate
   TimeSampleHandler

.. autofunction:: get_sampling_rate
.. autofunction:: set_sampling_rate
.. autoclass:: TimeSampleHandler
   :members:
"""

from __future__ import annotations
from numpy.typing import NDArray
from collections.abc import Callable

from functools import lru_cache
import numpy as np

try:
    import numba as nb

    has_numba = True
except:
    has_numba = False

from .constants import fs as fs_default

import warnings

if has_numba:
    from numba.core.errors import NumbaExperimentalFeatureWarning

    warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)


@lru_cache
def _ts(
    fs: int,
    nb_samples: int,
    n0: int = 0,
    sample_edges: bool = False,
    upsample_factor: int = 0,
):
    if upsample_factor > 0:
        P = 2**upsample_factor
        nb_samples *= P
        n0 *= P
        if sample_edges:
            n0 -= P - 1
            nb_samples += P - 1
        fs *= P
    elif sample_edges:
        nb_samples += 1
    t = np.arange(n0, n0 + nb_samples) / fs
    if sample_edges and not upsample_factor:
        t -= 0.5 / fs
    return t


class TimeSampleHandler:
    """Abstract base class of all the pyLeTalker classes"""

    _fs: int = fs_default  # sampling rate

    @classmethod
    def get_fs(cls) -> int:
        """get class-wide sampling rate"""
        return TimeSampleHandler._fs

    @classmethod
    def set_fs(cls, new_fs: int | None):
        """set class-wide sampling rate.

        :param new_fs: sampling rate in samples/second. If None, it
                       reverts the sampling rate to the default (44.1 kHz).
        """
        if new_fs is None:
            TimeSampleHandler._fs = fs_default
        else:
            try:
                new_fs = int(new_fs)
                assert new_fs > 0
            except (TypeError, AssertionError) as e:
                raise ValueError("fs must be a positive integer.") from e
            TimeSampleHandler._fs = new_fs

    @property
    def fs(self) -> int:
        """sampling rate in samples/second"""
        return TimeSampleHandler._fs

    @staticmethod
    def ts(
        nb_samples: int,
        n0: int = 0,
        sample_edges: bool = False,
        upsample_factor: int = 0,
    ) -> NDArray:
        """Return time vector in seconds.

        Parameters
        ----------
        nb_samples
            Number of samples
        n0, optional
            Starting time sample index, by default 0
        sample_edges, optional
            True to return time stamps associated with the edges of the samples, by default False
        upsample_factor, optional
            Specify a positive integer to increase the sampling rate of the returned time vector, by default 0
        """
        return _ts(TimeSampleHandler._fs, nb_samples, n0, sample_edges, upsample_factor)


def get_sampling_rate() -> int:
    """Return the global sampling rate in samples/second

    A common sampling rate is used for the entire pyLeTalker framework. As such,
    this method outputs the same value as :py:func:`letalker.fs` and
    :py:func:`letalker.get_sampling_rate()`.
    """
    return TimeSampleHandler.get_fs()


def set_sampling_rate(fs: int | None):
    """Sets the global sampling rate

    This function will change the common sampling rate across the entire
    pyLeTalker framework. It
    this method outputs the same value as :py:func:`letalker.fs` and
    :py:func:`letalker.get_sampling_rate()`.

    Parameters
    ----------
    fs
        New sampling rate in samples/second. Must be a positive value. If None,
        it reverts the sampling rate to the system default (44.1 khz)
    """
    TimeSampleHandler.set_fs(fs)


def ts(nb_samples: int, n0: int = 0):
    return TimeSampleHandler.ts(nb_samples, n0)


############################################################

_use_njit: bool = has_numba  # True to use Numba (default)


def use_numba(mode: bool):
    """Enable/disable Numba"""
    global _use_njit

    if mode and not has_numba:
        raise RuntimeError(
            "numba package is not available in the current Python environment."
        )

    _use_njit = bool(mode)


def using_numba() -> bool:
    """Returns True if Numba is enabled"""
    return _use_njit


def compile_njit_if_numba(func: Callable, **kwargs) -> Callable:
    return (nb.njit(**kwargs))(func) if _use_njit else func


def compile_jitclass_if_numba(
    CLS: type, spec: list[tuple[str, type]] = None, **kwargs
) -> type:
    return nb.experimental.jitclass(CLS, spec) if _use_njit else CLS


def compile_cfunc(func: Callable, c_sig: str, **kwargs) -> Callable:
    if _use_njit and "nopython" not in kwargs:
        kwargs["nopython"] = True
    return (nb.cfunc(c_sig, **kwargs))(func) if _use_njit else func


def _get_param(x: NDArray, i: int) -> float | NDArray:
    """return the i-th element or the last element of x"""
    return x[i] if i < x.shape[0] else x[-1]


from typing import TYPE_CHECKING

if TYPE_CHECKING:

    def get_param(x: NDArray, i: int) -> float | NDArray:
        """return the i-th element or the last element of x"""


def __getattr__(name):
    if name == "get_param":
        return nb.njit()(_get_param) if _use_njit else _get_param

    raise AttributeError()
