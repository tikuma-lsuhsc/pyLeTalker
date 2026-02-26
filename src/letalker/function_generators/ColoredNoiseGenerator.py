from __future__ import annotations

from typing import Literal
from numpy.typing import ArrayLike, NDArray
from typing_extensions import overload

from functools import cached_property
import numpy as np
from scipy.signal import (
    butter,
    tf2sos,
    sosfilt,
    sosfilt_zi,
    freqz,
    bilinear,
    find_peaks,
)

from .WhiteNoiseGenerator import WhiteNoiseGenerator, WhiteNoiseDistributionLiteral


class ColoredNoiseGenerator(WhiteNoiseGenerator):
    """Colored noise generator"""

    b: NDArray  #: numerator polynomial of the transfer function
    a: NDArray  #: denominator polynomial of the transfer function
    psd_level: float  #: peak value of power spectral density

    @overload
    def __init__(
        self,
        b: ArrayLike,
        a: ArrayLike | None = None,
        *,
        psd_level: float | None = None,
        innovation_distribution: WhiteNoiseDistributionLiteral = "gaussian",
    ):
        """constructor with filter coefficients

        Parameters
        ----------
        b
            numerator transfer function coefficients
        a
            denominator transfer function coefficients
        psd_level, optional
            PSD peak level in 1/Hz (default: 1)
        innovation_distribution, optional
            distribution of innovation process with zero mean and unit variance:
                "gaussian"   Gaussian (normal) distribution
                "uniform"    Uniform distribution
                "two_point"  Two-point (±1) distribution
        """

    @overload
    def __init__(
        self,
        N: int,
        Wp: float | ArrayLike,
        btype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
        *,
        psd_level: float | None = None,
        innovation_distribution: WhiteNoiseDistributionLiteral = "gaussian",
    ):
        """constructor using scipy.signal.butter

        Parameters
        ----------
        N
            The order of the filter. For 'bandpass' and 'bandstop' filters, the
            resulting order of the final second-order sections ('sos') matrix is
            2*N, with N the number of biquad sections of the desired system.
        ws
            The critical frequency or frequencies in Hz. For lowpass and highpass
            filters, Wn is a scalar; for bandpass and bandstop filters, Wn is a
            length-2 sequence. This is the point at which the gain drops to
            1/sqrt(2) that of the passband (the “-3 dB point”).
        btype, optional
            The type of filter. Default is 'lowpass'.
        psd_level, optional
            PSD peak level in 1/Hz (default: 1)
        innovation_distribution, optional
            distribution of innovation process with zero mean and unit variance:
                "gaussian"   Gaussian (normal) distribution
                "uniform"    Uniform distribution
                "two_point"  Two-point (±1) distribution
        """

    def __init__(
        self,
        *args,
        psd_level: float | None = None,
        innovation_distribution: WhiteNoiseDistributionLiteral = "gaussian",
        **kwargs,
    ):

        super().__init__(psd_level, innovation_distribution)

        nargs = len(args)
        nkwargs = len(kwargs)
        if nargs == 1 and nkwargs == 0:
            kwargs["b"] = args[0]
        elif nargs == 2 and nkwargs == 0 and not isinstance(args[0], int):
            kwargs["b"] = args[0]
            kwargs["a"] = args[1]

        if "b" in kwargs:
            b, a = np.asarray(kwargs["b"]), np.asarray(kwargs.get("a", [1]))
            # TODO - normalize b so the coloring filter has the passband gain of 1
        else:
            kwargs["output"] = "ba"
            if "analog" not in kwargs:
                kwargs["analog"] = False
            b, a = butter(*args, **kwargs, fs=self.fs)
            if kwargs.get("analog", True):
                b, a = bilinear(b, a, self.fs)

        self._b = b
        self._a = a

    @cached_property
    def sos(self) -> NDArray:
        """coefficients of the cascaded second-order sections of the noise coloring filter"""
        return tf2sos(self._b, self._a)

    @cached_property
    def z0(self) -> NDArray:
        """initial condition of the cascaded second-order sections of the noise coloring filter"""
        return np.zeros((self.sos.shape[0], 2))  # sosfilt_zi(self.sos)

    def generate(self, n: int, z0: ArrayLike | None = None) -> tuple[NDArray, NDArray]:
        """generate noise samples (no buffering)

        Parameters
        ----------
        n
            number of samples to generate
        z0, optional
            initial generator states, by default None

        Returns
        -------
            x
                generated samples
            z1
                final generator states, None if stateless generator
        """

        first = z0 is None or np.all(z0 == 0.0)
        n0 = self.transient_length if first else 0
        zi = self.z0 if z0 is None else np.array(z0)
        x, zo = sosfilt(self.sos, super().generate(n + n0)[0], zi=zi)
        return x[n0:], zo

    def psd(
        self,
        worN: int | ArrayLike | None = 512,
        whole: bool = False,
        plot: callable | None = None,
        include_nyquist: bool = True,
    ):
        """Compute the power spectral density

        Parameters
        ----------
        worN, optional
            If a single integer, then compute at that many frequencies (default is worN=512).
            If an array-like, compute the spectrum at the given frequencies in Hz.
        whole, optional
            True to compute frequencies from 0 to fs, by default False
        plot, optional
            A callable that takes two arguments. If given, the returned parameters w and
            h are passed to plot.
        include_nyquist, optional
            If `whole=False` and `worN` is an integer, setting `include_nyquist=True` will
            include the Nyquist frequency.

        Returns
        -------
            w: NDArray
                The frequencies in Hz at which `h` was computed
            h: NDArray
                The power spectrum
        """
        b = self.b * self.psd_level**0.5
        a = self.a
        f, H = freqz(b, a, worN, whole, plot, self.fs, include_nyquist)

        return f, np.abs(H) ** 2

    @cached_property
    def transient_length(self) -> int:
        """get the length of impulse response of the filter"""

        b, a = self._b, self._a
        if a.shape[0] == 1:
            return len(b)
        if np.any(np.abs(np.roots(a)) >= 1):
            return None  # unstable

        # the filter is stable, n is chosen as the point at which the term from the largest-amplitude pole is 5 × 10–5 times its original amplitude.
        sos = self.sos
        z0 = self.z0

        n0 = 0
        n = 1024
        x = np.zeros(n)
        x[0] = 1.0

        y, z0 = sosfilt(sos, x, zi=z0)

        while True:
            yabs = np.abs(y)

            ymax = yabs.max()
            yth = ymax * 1e-6

            pks = find_peaks(yabs)[0]
            if len(pks):
                # local maxima found
                j = np.argmax(yabs[pks] < yth)
                if j or np.all(yabs[pks[j] :] < yth):
                    if j:
                        return (
                            n0 + pks[j - 1] + np.argmax(yabs[pks[j - 1] : pks[j]] < yth)
                        )
                    else:
                        return n0 + pks[j] + np.argmax(yabs[: pks[j]] < yth)
            elif yabs[-1] <= yth:
                # no local maxima, assume monotonically decreasing
                return n0 + np.argmin(yabs < yth)

            n *= 2

            if n > 1e6:
                return None

            y, z0 = sosfilt(sos, np.zeros(n), zi=z0)
