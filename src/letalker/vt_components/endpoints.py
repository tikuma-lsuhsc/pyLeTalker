from typing import Any, Protocol, cast

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import rho_air as rho_air_default

rhoc_default = rho_air_default * c_default


class LTISinkFactory(Protocol):
    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.LTI:
        """Create a sink model (e.g., mouth/nose radiation)

        Parameters
        ----------
        area
            cross-sectional area in cm² of the tube section
            to be terminated by this sink.
        fs, optional
            sampling rate in samples/second to create a discrete-time model,
            by default the created model will be a continuous-time model.
        sample_kws, optional
            Keyword arguments to run ``ct.sample()`` function to discretize
            the model, by default the default parameters of ``ct.sample()`` will
            be used.

        Returns
        -------
            State-spece model with a single input :math:`F`,
            and at least one output, which is the reflected partial pressure
            :math:`B` . It may have additional outputs (or inputs).
        """


class LTISourceFactory(Protocol):
    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.LTI:
        """Create a source model (e.g., lungs)

        Parameters
        ----------
        area
            cross-sectional area in cm² of the tube section
            to be terminated by this source.
        fs, optional
            sampling rate in samples/second to create a discrete-time model,
            by default the created model will be a continuous-time model.
        sample_kws, optional
            Keyword arguments to run ``ct.sample()`` function to discretize
            the model, by default the default parameters of ``ct.sample()`` will
            be used.

        Returns
        -------
            State-spece model with a single output :math:`F`,
            and at least one input, which is the backward partial pressure
            :math:`B` . It may have additional inputs (or outputs).
        """


class LeTalkerLungs(LTISourceFactory):
    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.StateSpace:

        return ct.StateSpace(
            np.zeros([0, 0]),
            np.zeros([0, 2]),
            np.zeros([1, 0], np.array([[-0.8, 0.9]])),
            inputs=["PL", "B1"],
            outputs=["F1"],
        )


class FlanaganRadiationLoad(LTISinkFactory):
    """Flanagan's a-piston-in-a-infinite-buffle radiation model"""

    rho: float = rho_air_default
    c: float = c_default

    def __init__(self, rho: float | None = None, c: float | None = None):
        """Create a factory to create a default yielding wall transfer function model

        The transfer function is evaluated on the fly as the object is called
        with area and length of the vocal tract segment.

        Parameters
        ----------
        rho, optional
            the product of mass per unit surface area in g/cm²,
            by default 0.00114*35000
        c, optional
            speed of sound in cm/s, by default 35000
        """
        super().__init__()

        if rho is not None:
            self.rho = rho
        if c is not None:
            self.c = c

    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.TransferFunction:
        """Create a tf model of one vocal tract segment

        Parameters
        ----------
        area
            vocal tract segment cross-sectional area in cm²
        length
            vocal tract segment length in cm

        Returns
        -------
            Transfer function object which relates the radiated pressure
            output and the flow rate input (i.e., impedance)

        """

        a: float = (area / np.pi) ** 0.5
        Zm: float = self.rho * self.c / area

        three_pi: float = 3 * np.pi
        R: float = 128 * Zm / three_pi**2
        L: float = 8 * a * Zm / (three_pi * self.c)

        tf: ct.TransferFunction = cast(ct.TransferFunction, ct.tf([R * L, 0], [L, R]))

        return (
            tf
            if fs is None
            else tf.sample(1 / fs, **(sample_kws if sample_kws else {}))
        )


class TwoPortAcousticRadiator(LTISinkFactory):
    rhoc: float = rhoc_default
    u_to_rad: LTISinkFactory = FlanaganRadiationLoad()

    def __init__(
        self, rad_load: LTISinkFactory | None = None, rhoc: float | None = None
    ):
        """Two-port radiator based on a radiation impedance (a U->P system)

        Parameters
        ----------
        rad_load, optional
            radiation load system factory, by default FlanaganRadiationLoad is used
        rhoc, optional
            _description_, by default None
        """
        if rad_load is not None:
            self.u_to_rad = rad_load
        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.StateSpace:
        """Two-port reflective version of Flanagan's model with a piston in an infinite baffle

        Parameters
        ----------
        area
            _description_
        fs, optional
            _description_, by default None
        sample_kws, optional
            _description_, by default None

        Returns
        -------
            one-input (forward pressure)/two-output (radiated pressure & backward pressure)

            The radiated pressure output replaces the standard forward output term.
        """

        ss = cast(
            ct.StateSpace,
            (
                self.u_to_rad(area)
                if sample_last
                else self.u_to_rad(area, fs=fs, sample_kws=sample_kws)
            ).to_ss(),
        )

        assert ss.ninputs == 1 and ss.noutputs == 1

        z = self.rhoc / area
        a = cast(float, ss.D[0, 0]) / z

        # IN: F1
        # OUT: Prad, B1

        den = a + 1
        c0 = ss.C / den
        d0 = (a - 1) / den

        C = np.tile(c0, (2, 1))
        D = np.array([[1 + d0], [d0]])
        A = ss.A - ss.B @ C[1, :] / z
        B = ss.B * ((1 - D[1, 0]) / z)

        sys = cast(
            ct.StateSpace,
            ct.ss(A, B, C, D, dt=ss.dt, inputs=["F1"], outputs=["Prad", "B1"]),
        )

        if sample_last and fs is not None:
            sys = sys.sample(1 / fs, **(sample_kws or {}))

        return sys


class TwoPortStoryRadiator(LTISinkFactory):
    rhoc: float = rhoc_default
    u_to_rad: LTISinkFactory = FlanaganRadiationLoad()

    def __init__(self, sys: LTISinkFactory | None = None, rhoc: float | None = None):

        if sys is not None:
            self.u_to_rad = sys
        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.TransferFunction:
        """Discrete-time two-port reflective version of Flanagan's model with a piston in an infinite baffle

        Parameters
        ----------
        area
            _description_
        fs, optional
            _description_, by default None
        sample_kws, optional
            _description_, by default None

        Returns
        -------
            one-input (forward pressure)/two-output (radiated pressure & backward pressure)

            The radiated pressure output replaces the standard forward output term.
        """
        ss = cast(ct.TransferFunction, self.u_to_rad(area))
        assert ss.ninputs == 1 and ss.noutputs == 1
        assert fs is not None

        L, R = ss.den[0][0]

        z = self.rhoc / area

        Rp = R / z
        Lp = 2 * fs * L / z
        RLp = Rp * Lp

        # TF coefficients
        a2 = -Rp - Lp + RLp
        a1 = -Rp + Lp - RLp
        b2 = Rp + Lp + RLp
        b1 = -Rp + Lp + RLp

        B1num = [a2 / b2, a1 / b2]
        B1den = [1, -b1 / b2]
        Pnum = [(b2 + a2) / b2, (a1 - b1) / b2]
        Pden = [1, -b1 / b2]

        # [P;B1] <- [F1]
        return ct.tf(
            [[Pnum], [B1num]],
            [[Pden], [B1den]],
            1 / fs,
            inputs=["F1"],
            outputs=["Prad", "B1"],
        )


class VFFlowSource(LTISourceFactory):
    """Superior face of vocal folds interfacing the the first segment of the vocal tract

    Parameters
    ----------
    LTISourceFactory
        _description_

    Returns
    -------
        _description_
    """

    rhoc: float = rhoc_default
    r: float = 1.0

    def __init__(self, r: float | None = None, rhoc: float | None = None):

        if rhoc is not None:
            self.rhoc = rhoc
        if r is not None:
            self.r = r

    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
        ct_returnn_delay: float = 1e-9,
    ) -> ct.StateSpace:

        if fs is None:
            ...

        return ct.StateSpace(
            np.zeros([0, 0]),
            np.zeros([0, 2]),
            np.zeros([1, 0]),
            np.array([[self.rhoc / area, self.r]]),
            fs and 1 / fs,
            inputs=["Ug", "B2"],
            outputs=["F2"],
        )

    def terminate(self, sys: ct.LTI, area: float) -> ct.StateSpace:
        ss = sys.to_ss()
        Q = np.array([[self.rhoc / area, 1.0], [0, 1]])  # F1/B2 => Ug/B2
        ss.B = ss.B @ Q
        (d11, d12), (d21, d22) = ss.D
        # g1 =
        # g2 = 1.0
        Q = np.array([[1, -d11 * g2], [0, 1 - d21 * g2]])


class VFFlowSink(LTISinkFactory):
    """Inferior face of vocal folds interfacing the the first segment of the vocal tract

    Parameters
    ----------
    LTISourceFactory
        _description_

    Returns
    -------
        _description_
    """

    rhoc: float = rhoc_default

    def __init__(self, rhoc: float | None = None):

        if rhoc is not None:
            self.rhoc = rhoc

    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.StateSpace:

        # input: F1, Ug
        # output: B2

        return ct.StateSpace(
            np.zeros([0, 0]),
            np.zeros([0, 2]),
            np.zeros([1, 0]),
            np.array([[1.0, -self.rhoc / area]]),
            fs and 1 / fs,
            inputs=["F1"],
            outputs=["Ug", "B1"],
        )


class ResistiveLoadSink(LTISinkFactory):
    def __call__(
        self,
        area: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.StateSpace:

        # INPUT: F1
        # OUTPUT: P, B1

        r = 0.9

        return ct.StateSpace(
            np.zeros([0, 0]),
            np.zeros([0, 1]),
            np.zeros([2, 0]),
            np.array([[1 + r], [r]]),
            fs and 1 / fs,
            inputs=["F1"],
            outputs=["B1"],
        )
