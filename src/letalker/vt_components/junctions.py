from typing import Any, Protocol

import control as ct
import numpy as np

from ..constants import c as c_default
from ..constants import rho_air as rho_air_default

rhoc_default = rho_air_default * c_default


class LTIJunctionFactory(Protocol):
    def __call__(
        self,
        *areas: tuple[float],
        nb_inputs: int = 1,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
    ) -> ct.StateSpace:
        """create a feed-through only two-port junction system

        Parameters
        ----------
        *areas : float
            cross-sectional areas in cm² of the sections that are
            joining at the junction.
        nb_inlets, optional
            The first `nb_inlets` (by default 1) areas are interpreted 
            as the input sections, which positive flow flows into the 
            junction. The remaining areas are the areas of the output 
            sections, which positive flow flows out of the junction.
        fs, optional
            sampling rate in samples/second to create a discrete-time
            model, by default None
        sample_kws, optional
            (not used) discretization keyword options, by default None

        Returns
        -------
            feed-through only state-space model

            .. math::

                \mathbf{p}_{out} = \mathbf{D} \mathbf{p}_{in}

            where :math:`\mathbf{D}` is the feedthrough matrix and
            the input partial pressure vector 
            
            .. math::
            
                \mathbf{p}_{in} = \begin{bmatrix}
                F_1\\F_2\\\vdots\\F_{N_\text{in}}\\B_{N_\text{in}+1}\\B_{N_\text{in}+2}\\\vdots\\B_{N}\\
                \end{bmatrix}

            and the output partial pressure vector

            .. math::
            
                \mathbf{p}_{out} = \begin{bmatrix}
                F_{N_\text{in}+1}\\F_{N_\text{in}+2}\\\vdots\\F_{N}\\B_1\\B_2\\\vdots\\B_{N_\text{in}}\\
                \end{bmatrix}

            Here, $N$ is the number values in `areas` and $N_\text{in}$ = `nb_inlets`.
            Note that the input and output partial pressures are always ordered with 
            the forward flows first.

        """


class LosslessJunction(LTIJunctionFactory):
    """lossless tube segment junction"""

    def __call__(
        self,
        *areas: tuple[float],
        nb_inlets: int = 1,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        _force_mimo: bool=False
    ) -> ct.StateSpace:

        nsegs = len(areas)

        if not _force_mimo and nsegs == 2 and nb_inlets == 1:
            return self._siso(*areas, fs)

        # compute the admittances of each joining segment
        y = np.array(areas).reshape((1, -1))

        nin = nb_inlets - 1  # num inlets minus the first
        nout = nsegs - nb_inlets  # num outlets

        A = np.block(
            [
                [y[:, nb_inlets:], y[:, :nb_inlets]],
                [np.zeros((nin, nout)), np.full((nin, 1), -1), np.eye(nin)],
                [np.eye(nout), np.full((nout, 1), -1), np.zeros((nout, nin))],
            ]
        )
        B = np.block([[y], [np.ones((nsegs - 1, 1)), -1 * np.eye(nsegs - 1)]])

        return ct.ss(
            np.empty((0, 0)),
            np.empty((0, 2 * nb_inlets)),
            np.empty((2 * nout, 0)),
            np.linalg.lstsq(A, B)[0],
            dt=fs and 1 / fs,
        )

    def _siso(self, area1: float, area2: float, fs: float | None) -> ct.StateSpace:

        den = area1 + area2
        m1 = area1 / den
        m2 = area2 / den
        d = m1 - m2

        D = np.array([[2 * area1 / den, -d], [d, 2 * area2 / den]])

        return ct.ss(
            np.empty((0, 0)), np.empty((0, 2)), np.empty((2, 0)), D, dt=fs and 1 / fs
        )
