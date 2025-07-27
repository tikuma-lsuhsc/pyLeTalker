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
from numpy.typing import NDArray, ArrayLike
from fractions import Fraction
import numpy as np

from math import ceil
from functools import reduce

from ..function_generators.abc import FunctionGenerator
from ..constants import fs as default_fs, c, rho_air as rho
from ..function_generators import Interpolator, Constant

_pi = np.pi
_2pi = 2 * _pi
_j2pi = 2j * _pi

__all__ = ["strobe_timing"]


def strobe_timing(
    fr: int | Fraction,
    fo: FunctionGenerator | Sequence[float] | float,
    duration: float | None = None,
    *,
    fstrobe: int | Fraction = Fraction(6, 5),
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


def vt_areas_to_tf(
    f: ArrayLike,
    areas: ArrayLike,
    delta_l: float,
    a: float | None = None,
    Fw: float | None = None,
    c1: float | None = None,
    FT: float | None = None,
    c: float = c,
    rho: float = rho,
) -> NDArray:
    """Compute the 2-in 2-out frequency response of vocal tract

    Parameters
    ----------
    f
        Length-N frequency vector in Hz
    areas
        vocal tract cross-sectional area vector in cm², ordered from glottis to lips
    delta_l
        length of each vocal tract tublet in cm
    a, optional
        ratio of wall resistance to mass in rad/s, by default None to use 130π
    Fw, optional
        frequency of mechanical resonance in Hz, by default None
        to use 15 Hz
    c1, optional
        correction for thermal conductivity and viscosity in rad/s, by default
        None to use 4
    FT, optional
        lowest resonant frequency of closed tract in Hz, by default
        None to use 203 Hz
    c, optional
        speed of sound in cm/s, by default letalker.constants.c (35000)
    rho, optional
        air density in g/cm^3 (= kg/mm^3), by default letalker.constants.rho_air
        (0.00114)

    Returns
    -------
    K
        The Nx2x2 NumPy array of complex values of the vocal tract's frequency
        response estimated at each of the frequency values given in f:

            [Pout(f);Uout(f)] = K(f) [Pin(f);Uin(f)]

        Pin and Pout are the pressure at the input (glottis) and output (lips),
        respectively; and Uin and Uout are the flows at the respective ends of
        the vocal folds

    References
    ----------
    [1] M. Sondhi and J. Schroeter, “A hybrid time-frequency domain articulatory
        speech synthesizer,” IEEE Trans. Acoust., Speech, Signal Process.,
        vol. 35, no. 7, pp. 955–967, July 1987, doi: 10.1109/TASSP.1987.1165240.
    [2] B. H. Story, A.-M. Laukkanen, and I. R. Titze, “Acoustic impedance of an
        artificially lengthened and constricted vocal tract,” Journal of Voice,
        vol. 14, no. 4, pp. 455–469, Dec. 2000, doi: 10.1016/S0892-1997(00)80003-X.

    """

    if c1 is None:
        c1 = 4
    if a is None:
        a = 130 * _pi
    if Fw is None:
        Fw = 15
    if FT is None:
        FT = 203

    jomega = _j2pi * np.asarray(f)
    omega0 = _2pi * FT
    b = Fw * _2pi
    alpha = np.sqrt(jomega * c1)
    beta = jomega * omega0**2 / ((jomega + a) * jomega + b**2) + alpha
    gamma = np.sqrt((a + jomega) / (beta + jomega))
    sigma = gamma * (beta + jomega)
    sigdl_c = sigma * delta_l / c

    areas = np.asarray(areas).reshape(-1, 1)

    ABCD = np.empty((areas.size, jomega.size, 2, 2), dtype=complex)
    ABCD[:, :, 0, 0] = ABCD[:, :, 1, 1] = np.cosh(sigdl_c)
    BC1 = -rho * c / areas * gamma
    BC2 = np.sinh(sigdl_c)
    ABCD[:, :, 0, 1] = BC1 * BC2
    ABCD[:, :, 1, 0] = BC2 / BC1

    return reduce(np.matmul, ABCD[1:], ABCD[0])


def lips_z_load(f: ArrayLike, a: float, c: float = c) -> NDArray:
    """calculate radiation impedance

    Parameters
    ----------
    f
        Length-N frequency vector in Hz
    a
        cross-sectional area between lips
    c, optional
        speed of sound, by default letalker.constants.c (35000)

    Returns
    -------
        Length-N radiation impedance

    References
    ----------
    [1] B. H. Story, A.-M. Laukkanen, and I. R. Titze, “Acoustic impedance of an
        artificially lengthened and constricted vocal tract,” Journal of Voice,
        vol. 14, no. 4, pp. 455–469, Dec. 2000, doi: 10.1016/S0892-1997(00)80003-X.
    """

    jomega = _j2pi * np.asarray(f)
    bm = np.sqrt(a / _pi)
    Zm = rho * c / a
    R = 128 * Zm / (9 * _pi**2)
    L = 8 * bm * Zm / (3 * _pi * c)
    return jomega * R * L / (R + jomega * L)


def vt_lips_z_in(
    f: ArrayLike,
    areas: ArrayLike,
    delta_l: float,
    a: float | None = None,
    b: float | None = None,
    c1: float | None = None,
    omega0_2: float | None = None,
    c: float = c,
    rho: float = rho,
) -> NDArray:
    """Compute input impedance of vocal tract with lip radiation

    Parameters
    ----------
    f
        Length-N frequency vector in Hz
    areas
        vocal tract cross-sectional area vector in cm², ordered from glottis to lips
    delta_l
        length of each vocal tract tublet in cm
    a, optional
        ratio of wall resistance to mass in rad/s, by default None to use 130π
    Fw, optional
        frequency of mechanical resonance in Hz, by default None
        to use 15 Hz
    c1, optional
        correction for thermal conductivity and viscosity in rad/s, by default
        None to use 4
    FT, optional
        lowest resonant frequency of closed tract in Hz, by default
        None to use 203 Hz
    c, optional
        speed of sound in cm/s, by default letalker.constants.c (35000)
    rho, optional
        air density in g/cm^3 (= kg/mm^3), by default letalker.constants.rho_air
        (0.00114)

    Returns
    -------
    K
        The Nx2x2 NumPy array of complex values of the vocal tract's frequency
        response estimated at each of the frequency values given in f:

            [Pout(f);Uout(f)] = K(f) [Pin(f);Uin(f)]

        Pin and Pout are the pressure at the input (glottis) and output (lips),
        respectively; and Uin and Uout are the flows at the respective ends of
        the vocal folds

    References
    ----------
    [1] M. Sondhi and J. Schroeter, “A hybrid time-frequency domain articulatory
        speech synthesizer,” IEEE Trans. Acoust., Speech, Signal Process.,
        vol. 35, no. 7, pp. 955–967, July 1987, doi: 10.1109/TASSP.1987.1165240.
    [2] B. H. Story, A.-M. Laukkanen, and I. R. Titze, “Acoustic impedance of an
        artificially lengthened and constricted vocal tract,” Journal of Voice,
        vol. 14, no. 4, pp. 455–469, Dec. 2000, doi: 10.1016/S0892-1997(00)80003-X.
    """
    K = vt_areas_to_tf(f, areas, delta_l, a, b, c1, omega0_2, c, rho)
    ZL = lips_z_load(f, areas[-1], c)
    return (K[..., 1, 1] * ZL - K[..., 0, 1]) / (K[..., 0, 0] - K[..., 1, 0] * ZL)
