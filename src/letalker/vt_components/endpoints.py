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
    ) -> ct.StateSpace:

        return ct.StateSpace(
            np.zeros([0, 0]),
            np.zeros([0, 2]),
            np.zeros([1, 0], np.array([[-0.8, 0.9]])),
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


class TwoPortFlanaganRadiator(LTISinkFactory):
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
        ss = cast(ct.StateSpace, self.u_to_rad(area).to_ss())
        assert ss.ninputs == 1 and ss.noutputs == 1

        z = self.rhoc / area
        dz = cast(float, ss.D[0, 0]) / z

        Q = np.array([[1 + dz, 0], [dz, 1]])
        C = np.linalg.lstsq(Q, np.full((2, 1), ss.C))[0]
        D = np.linalg.lstsq(Q, np.array([[dz - 1], [dz]]))[0]
        A = ss.A - ss.B @ C[0, :]
        B = ss.B * (1 / z - D[0, 0])

        sys = cast(ct.StateSpace, ct.ss(A, B, C, D))

        if fs is not None:
            sys = sys.sample(1 / fs, **(sample_kws or {}))

        return sys


class VFFlowSource(LTISourceFactory):
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
    ) -> ct.StateSpace:

        return ct.StateSpace(
            np.zeros([0, 0]),
            np.zeros([0, 2]),
            np.zeros([1, 0]),
            np.array([[self.rhoc / area, 0.9]]),
            fs and 1 / fs,
        )
