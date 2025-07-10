"""Klatt Aspiration Noise Injector

Klatt & Klatt describes their aspiration noise as [1, p.838]:

"The aspiration noise has a spectrum that falls off at about 6 dB/oct of frequency 
increase, but when the radiation characteristic is folded into the source models, 
the result is a relatively flat spectrum. Thus, as noise is added to the voicing
source, it is most noticeable at high frequencies where the harmonic spectrum of 
voicing is weaker. If the glottis is partially spread, as in a breathy vowel, TL 
and AH will be increased, and higher harmonics of the source spectrum will be 
replaced by aspiration noise."

According to `klatt.c` of the eSpeak synthesizer [2], the 3-dB cut-off frequency 
of the lowpass process is around 4000 Hz (a discrete-time AR(1) process with pole 
at 0.75, sampled at 88.2 kS/s).

Also in [2], the noise power is modulated during phonation: the amplitude is halved
for the half (closing phase) of the source vibration period.

Changes in KlattAspirationNoise:

1. The default lowpass filter is changed to a bilinear transformed first-order 
Butterworth lowpass filter with the 4 kHz cutoff.

2. The modulation is imposed by Reynolds number, a la `KlattAspirationNoise`.
Unlike `KlattAspirationNoise`, noise remain present during sub-critical period
with the reduction factor `alpha`.

[1] D. H. Klatt and L. C. Klatt, “Analysis, synthesis, and perception of voice quality variations among female and male talkers,” J. Acoust. Soc. Am., vol. 87, no. 2, pp. 820–857, 1990, doi: 10.1121/1.398894.
[2] D. H. Klatt, J. Iles, N. Ing-Simmons, Reece H. Dunn, https://github.com/espeak-ng/klatt/blob/master/parwave.c

"""

# /*
#    function GEN_NOISE

#    Random number generator (return a number between -8191 and +8191)
#    Noise spectrum is tilted down by soft low-pass filter having a pole near
#    the origin in the z-plane, i.e. output = input + (0.75 * lastoutput)
#  */

# static double gen_noise(double noise)
# {
# 	long temp;
# 	static double nlast;

# 	temp = (long)getrandom(-8191, 8191);
# 	kt_globals.nrand = (long)temp;

# 	noise = kt_globals.nrand + (0.75 * nlast);
# 	nlast = noise;

# 	return noise;
# }

# // Amplitude modulate noise (reduce noise amplitude during
# // second half of glottal period) if voicing simultaneously present.

# if (kt_globals.nper > kt_globals.nmod)
# 	noise *= (double)0.5;

# // Compute frication noise
# frics = kt_globals.amp_frica * noise;

# 22050*4

from __future__ import annotations

from numpy.typing import NDArray

from dataclasses import dataclass
import numpy as np

from ..__util import format_parameter
from ..function_generators import ColoredNoiseGenerator
from ..function_generators.abc import NoiseGenerator

from .abc import AspirationNoise, Element

from ..core import has_numba
if has_numba:
    import numba as nb


class KlattAspirationNoise(AspirationNoise):

    alpha: float = 0.5
    """sub-critical noise level"""
    REc: float = 1200.0
    """Critical Reynolds number for full-strength noise injection"""

    @dataclass
    class Results(Element.Results):
        src_noise: NDArray
        re2b: NDArray
        re2: NDArray
        unoise: NDArray

        @property
        def re(self) -> NDArray:
            return self.re2**0.5

        @property
        def re_b(self) -> NDArray:
            return self.re2b**0.5

    # override result class
    _ResultsClass = Results

    RunnerSpec = [
        ("n", nb.int64),
        ("nuL_inv", nb.float64[:]),
        ("nf", nb.float64[:]),
        ("alpha", nb.float64),
        ("re2b", nb.float64[:]),
        ("re2", nb.float64[:]),
        ("ug_noise", nb.float64[:]),
    ] if has_numba else []

    class Runner:
        n: int
        nuL_inv: NDArray
        nf: NDArray
        alpha: NDArray
        re2b: NDArray
        re2: NDArray
        ug_noise: NDArray

        def __init__(
            self,
            nb_steps: int,
            s: NDArray,
            nuL_inv: NDArray,
            nf: NDArray,
            alpha: NDArray,
            RE2b: NDArray,
        ):

            self.n = nb_steps
            self.nuL_inv = nuL_inv
            self.nf = nf
            self.alpha = alpha
            self.re2b = RE2b
            self.re2 = np.empty(nb_steps)
            self.ug_noise = np.empty(nb_steps)

        def step(self, i: int, ug: float) -> float:

            nuL_inv = self.nuL_inv[i if i < self.nuL_inv.shape[0] else -1]
            nf = self.nf[i if i < self.nf.shape[0] else -1]
            alpha = self.alpha
            RE2b = self.re2b[i if i < self.re2b.shape[0] else -1]
            self.re2[i] = RE2 = (ug * nuL_inv) ** 2.0

            self.ug_noise[i] = u = nf if RE2 > RE2b else alpha * nf
            # self.ug_noise[i] = u = (
            #     ((RE2 - RE2b) * nf) if RE2 > RE2b else (RE2b * alpha * nf)
            # )

            return u

    def __init__(
        self,
        noise_source: NoiseGenerator | float | dict | None = None,
        alpha: float | None = None,
        REc: float | None = None,
    ):
        """Noise generator in Klatt1990 synthesizer

        Parameters
        ----------
        noise_source, optional
            Aspiration noise source. If float value is given, ColoredNoiseGenerator
            is used to generate 1st-order lowpass Gaussian noise with -6 dB/oct rolloff with
            the specified DC level in (cm³/s)/Hz, by default (None) the level is set to 200
            (cm³/s)/Hz
        alpha, optional
            Noise level loss relative to the default level in (cm³/s)/Hz during sub-critical period
            (including the closed-phase period), by default None to use 0.5
        REc, optional
            Critical Reynolds number to produce the full aspiration noise, by default None
        """
        super().__init__()

        if REc is not None:
            self.REc = REc

        self.RE2b = self.REc**2

        create_noise_source = not isinstance(noise_source, NoiseGenerator)
        if create_noise_source:
            # default peak noise level in volumetric flow rate (cm³/s)/Hz
            noise_level: float = 5

            is_dict = isinstance(noise_source, dict)

            noise_source_kws = noise_source if is_dict else {"b": 0.25, "a": [1, -0.75]}

            if not is_dict and noise_source is not None:
                noise_level = noise_source

            if "psd_level" not in noise_source_kws:
                noise_source_kws["psd_level"] = noise_level**2

        self.noise_source = (
            ColoredNoiseGenerator(**noise_source_kws)
            if create_noise_source
            else noise_source
        )
        self.alpha = float(alpha or 0.5)

        self.nf = None

    @property
    def need_length(self) -> bool:
        """True if vocal fold length is necessary to generate aspiration noise"""
        return True

    def generate_noise(self, nb_samples: int, n0: int = 0) -> NDArray:
        return self.noise_source(nb_samples, n0)

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:
        return KlattAspirationNoise.Runner, KlattAspirationNoise.RunnerSpec

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "nf", "re2b", "re2", "ug_noise"]

    def generate_sim_params(
        self, n: int, n0: int = 0, *, length: NDArray
    ) -> tuple[NDArray, ...]:
        """Generate element parameters (possibly time varying)"""

        nuL_inv = 1 / (self.nu * length)
        nf = self.generate_noise(n, n0)
        alpha = self.alpha

        RE2b = float(self.RE2b)

        return np.atleast_1d(nuL_inv), np.atleast_1d(nf), alpha, np.atleast_1d(RE2b)
