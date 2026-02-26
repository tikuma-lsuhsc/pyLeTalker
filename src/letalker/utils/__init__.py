"""Utility Functions

The :py:mod:`letalker.utils` module defines utility functions to complement the
synthesis functionality of pyLeTalker.

.. autosummary::
   :nosignatures:

   strobe_timing


.. autoclass:: strobe_timing


"""

from __future__ import annotations

from typing import Sequence
from numpy.typing import NDArray
from fractions import Fraction
import numpy as np

from math import ceil

from ..function_generators.abc import FunctionGenerator
from ..constants import fs as default_fs
from ..function_generators import Interpolator, Constant


__all__ = ["strobe_timing"]


def strobe_timing(
    fr: int | Fraction,
    fo: FunctionGenerator | Sequence[float] | float,
    duration: float | None = None,
    *,
    fstrobe: int | Fraction = 1.2,
    n0: int = 0,
) -> NDArray:
    """compute non-uniform sampling indices for the strobe effect

    Parameters
    ----------
    fr
        framerate of the video in frames/second
    fo
        Either a function generator or a sequence of the fundamental
        frequency at the simulation sampling rate or a float if fo is
        fixed. Fundamental frequency is specified in Hz.
        TODO: If a  periodicity is lost at any point, the fundamental
        frequency may be <=0 or `nan`.
    duration, optional
        duration of the video in seconds. Required if `fo` is not a
        sequence, by default None
    fstrobe, optional
        strobe rate, the number of cycles to show per second, by default 2
    n0, optional
        starting time for `fun`, by default 0

    Returns
    -------
        time indices of the output samples of `fun`
    """

    nb_samples = None

    if isinstance(fo, FunctionGenerator):
        fs = fo.fs
    else:
        fs = default_fs

        # if fo is float or a seq of floats, convert to function generator
        try:
            fo = np.asarray(fo, dtype=float)
        except ValueError as e:
            raise ValueError("fo must be a FunctionGenerator object or float ")

        if fo.ndim:
            if fo.ndim > 1:
                raise ValueError("fo must be 1D sequence")

            if fo.shape[0] == 1:
                fo = fo[0]
            else:
                if duration is None:
                    nb_samples = len(fo)
                fo = Interpolator(fs, fo, degree=min(fo.shape[0] - 1, 3))

        if not (isinstance(fo, FunctionGenerator) or fo.ndim):
            fo = Constant(fo)

    if nb_samples is None:
        if duration is None:
            raise TypeError(
                "duration must be specified if fo is not specified as a sequence."
            )

    nb_samples = round(duration * fs)  # number of audio samples
    nframes = ceil(duration * fr)  # number of video frames

    # wrapped normalized phase function
    phi = fo.antiderivative(nb_samples, n0)
    phi_wrapped = (phi - phi[0]) % 1  # make sure the first sample is 0

    # desired normalized phase to capture by each frame
    phi_n = ((np.arange(nframes) / fr) * fstrobe) % 1.0

    # boundary sample indices between successive frames
    nedges = np.round((0.5 + np.arange(nframes)) * (fs / fr)).astype(int)

    # preallocate the output array
    nout = np.empty(nframes, dtype=int)
    nout[0] = 0  # first frame is always at 0

    for i, (phi_i, ni0, ni1) in enumerate(zip(phi_n[1:], nedges[:-1], nedges[1:])):
        # find the rising edges during the frame duration
        tf = phi_wrapped[ni0:ni1] > phi_i
        j = np.where(~tf[:-1] & tf[1:])[0]
        falling_edge = not len(j)

        if falling_edge:  # no rising edges found
            # no rising edge -> extreme ends, close to either 0 or 1, look for the falling edges instead
            tf = phi_wrapped[ni0:ni1] > 0.5
            j = np.where(tf[:-1] & ~tf[1:])[0]

        # select the rising edge closest to the center of the frame duration
        jc = (ni1 - ni0) / 2  # center frame position
        nsel = j[np.argmin(np.abs(j + 0.5 - jc))] + ni0

        # select the sample of the selected edge with the phase closest to the desired
        if falling_edge:
            if phi_i < 0.5:
                phi_i += 1.0
            # phi0 = phi_wrapped[nsel]
            # phi1 = phi_wrapped[nsel + 1] + 1.0
            # if phi1 - phi_i < phi_i - phi0:
            if phi_wrapped[[nsel + 1, nsel]].sum() < 2 * phi_i - 1.0:
                nsel += 1
        elif np.argmin(np.abs(phi_wrapped[[nsel, nsel + 1]] - phi_i)):
            nsel += 1

        # assign the index
        nout[i + 1] = nsel

    return nout
