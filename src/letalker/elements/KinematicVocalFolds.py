from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, partial
from math import pi
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import lambertw
from typing_extensions import overload

_2pi = 2 * pi

from .. import constants
from ..__util import format_parameter
from ..constants import (
    kinematic_params as kinematic_params_default,
)
from ..constants import (
    rho_tis as default_rho_tis,
)
from ..core import has_numba
from ..errors import LeTalkerError
from ..function_generators import Constant, PeriodicInterpolator, SineGenerator
from ..function_generators.abc import (
    AnalyticFunctionGenerator,
    FunctionGenerator,
    SampleGenerator,
)
from .abc import AspirationNoise, Element, VocalTract
from .VocalFoldsAg import VocalFoldsAgBase

if has_numba:
    import numba as nb

Fo2L_Callable = Callable[[float, float], float]
"""convert fundamental frequency to glottal length

    Parameters
    ----------
    fo
        target fundamental frequency in Hz (=1/s)
    L0
        relaxed glottal length in cm (1.6 for male, 1.0 for female)

    Returns
    -------
        glottal length in mm

"""


def fo_to_L_LeTalker(fo: float, L0: float, fo_n: float = 125) -> float:
    return 0.56 * L0 * np.sqrt((fo / fo_n))


def fo_to_L(
    fo: float, L0: float, A: float, B: float, rho: float = default_rho_tis
) -> float:
    """convert fundamental frequency to glottal length

    Parameters
    ----------
    fo
        target fundamental frequency in Hz (=1/s)
    L0
        relaxed glottal length in cm (1.6 for male, 1.0 for female)
    A
        stress-strain curve gain term in kPa (= kg/(mm-s^2)) (male: 4.0, female: 5.5)
    B
        stress-strain curve exponential term (male: 9.4, female: 7.6)
    rho, optional
        tissue density in g/cm^3 (= kg/mm^3), by default constants.default_rho_tis (1.04)

    Returns
    -------
        glottal length in mm

    Based on the high-strain exponential relationship of the length to fundamental frequency:

    fo = 1/(2L) sqrt(A/rho exp(B(L-L0)/L0)) # exponential
         1/(2L) sqrt((a(L-L0)/L0 + b)/rho)

    [1] I. Titze, T. Riede, and T. Mau, “Predicting achievable fundamental
        frequency ranges in vocalization across species,” PLoS Comput Biol,
        vol. 12, no. 6, p. e1004907, Jun. 2016, doi: 10.1371/journal.


    2 fo sqrt(rho/A) exp(B/2) = 1/L exp(B/(2L0) L)
    1/(2 fo sqrt(rho/A) exp(B/2)) = L exp(-B/(2L0) L)
    c b = c L exp(c L)

    L0: mm
    A: kPa
    rho:


    """

    b = 1 / (2 * fo * (rho / A) ** 0.5 * np.exp(B / 2))
    c = -B / (2 * L0 * 1e-2)
    x = b * c

    L = np.empty_like(x)
    tf = x > 0
    L[tf] = lambertw(x[tf]).real / c
    tf = ~tf
    L[tf] = lambertw(x[tf], -1).real / c

    return L * 1e2


def _validate_param(name: str, x: NDArray, ndim: int) -> NDArray:

    shape = (-1, *np.ones(ndim - 1))

    if x.ndim >= ndim:
        if x.ndim != ndim or x.shape[-1] != 2:
            raise LeTalkerError(
                f"Incompatible {name}. {name} must be or produce {ndim}D array for symmetric vocal fold model or "
                f"{ndim + 1}D array for left-right asymmetric vocal fold model with the last dimension being 2."
            )
        x = np.moveaxis(x, -1, 0).reshape(2, *shape)
    else:
        x = x.reshape(shape)

    return x


def _load_asym_param(
    x: SampleGenerator,
    ndim_sym: int,
    nb_samples: int | None,
    n0: int = 0,
    force_data_axis: bool = True,
    analytic: bool = False,
) -> NDArray:
    """generate samples and place the lateral axis to axis 0 if present

    Parameters
    ----------
    x
        sample generator
    ndim_sym
        expected ndim for fixed symmetric VFs (no lateral axis, no time axis)
    nb_samples
        number of samples to generate
    n0, optional
        first sample index, by default 0
    force_data_axis, optional
        True to append an axis if scalar
    analytic, optional
        True for periodic function generator to return analytic signal, by default False

    Returns
    -------
        generated sample array with axes: (lateral?, time, *data)

    """
    xvals = x(nb_samples or 1, n0, analytic=analytic)
    # guarantees

    has_lateral = x.ndim > ndim_sym and x.shape[0] > 1
    if has_lateral:  # lateral axis given
        # swap time and lateral axes
        xvals = xvals.reshape(2, 1) if xvals.ndim < 2 else np.moveaxis(xvals, 0, 1)

    if force_data_axis:
        # expected array size
        ndim = ndim_sym + (2 if has_lateral else 1)

        # add data axis if scalar
        while xvals.ndim < ndim:
            xvals = xvals[..., np.newaxis]

    return xvals


Xi0Z_Callable = Callable[[NDArray, int, int], NDArray]
"""calculate prephonatory posterior VF edge displacements at all z-samples

        Parameters
        ----------
        z
            normalized z-axis, 0=most inferior, 1=most superior
        nb_samples
            simulation duration in number of time samples
        n0
            simulation starting time

        Returns
        -------
        xi0
            prephonatory posterior VF edge at the depths 
            specified by z. The shape of the array differs depending on the configuration:

            ==========  ========  =========================
            symmetry    temporal  shape
            ==========  ========  =========================
            symmetric   fixed     z.shape
            symmetric   dynamic   (nb_samples, *z.shape)
            asymmetric  fixed     (2, *z.shape)
            asymmetric  dynamic   (2, nb_samples, *z.shape)
            ==========  ========  =========================
"""


class Xi0ZFunBase(Xi0Z_Callable, metaclass=abc.ABCMeta):
    """Abstract base of prephonatory posterior VF edge positioning functor"""

    @abc.abstractmethod
    def _get_xi0(self, nb_samples: int | None, n0: int = 0) -> NDArray:
        """calculate prephonatory posterior VF edge at most inferior & superior depths

        Parameters
        ----------
        nb_samples
            simulation duration in number of time samples
        n0, optional
            simulation starting time, by default 0

        Returns
        -------
        xi0
            prephonatory posterior VF edge at the most inferior & superior depths. The shape
            of the array differs depending on the configuration:

            ==========  ========  ==================
            symmetry    temporal  shape
            ==========  ========  ==================
            symmetric   fixed     (1, 2)
            symmetric   dynamic   (nb_samples, 2)
            asymmetric  fixed     (2, 1, 2)
            asymmetric  dynamic   (2, nb_samples, 2)
            ==========  ========  ==================

        """

    @abc.abstractmethod
    def __call__(self, z: NDArray, nb_samples: int | None, n0: int = 0) -> NDArray:
        """calculate prephonatory posterior VF edge at all z-samples

        Parameters
        ----------
        z
            normalized z-axis, 0=most inferior, 1=most superior
        nb_samples
            simulation duration in number of time samples
        n0, optional
            simulation starting time, by default 0

        Returns
        -------
        xi0
            prephonatory posterior VF edge at the depths specified by z. The shape
            of the array differs depending on the configuration:

            ==========  ========  =========================
            symmetry    temporal  shape
            ==========  ========  =========================
            symmetric   fixed     z.shape
            symmetric   dynamic   (nb_samples, *z.shape)
            asymmetric  fixed     (2, *z.shape)
            asymmetric  dynamic   (2, nb_samples, *z.shape)
            ==========  ========  =========================
        """


class Xi0ZFunPolyBase(Xi0ZFunBase):
    Qb: SampleGenerator | None  # bulging quotient

    def __init__(self, Qb: float | ArrayLike | SampleGenerator | None = None):
        """Abstract base of polynomial prephonatory posterior VF edge positioning functor

        Parameters
        ----------
        Qb, optional
            bulging quotient if xi0 specifies only 2 endpoints, by default None (no bulge)
        """
        self.Qb = format_parameter(
            Qb, optional=True, shape=[(), (1,), (2,)], force_time_axis=False
        )

    def __call__(self, z: NDArray, nb_samples: int | None, n0: int = 0) -> NDArray:
        """calculate prephonatory posterior VF edge at all z-samples

        Parameters
        ----------
        z
            normalized z-axis, 0=most inferior, 1=most superior
        nb_samples
            simulation duration in number of time samples
        n0, optional
            simulation starting time, by default 0

        Returns
        -------
        xi0
            prephonatory posterior VF edge at the depths specified by z. The shape
            of the array differs depending on the configuration:

            ==========  ========  =========================
            symmetry    temporal  shape
            ==========  ========  =========================
            symmetric   fixed     z.shape
            symmetric   dynamic   (nb_samples, *z.shape)
            asymmetric  fixed     (2, *z.shape)
            asymmetric  dynamic   (2, nb_samples, *z.shape)
            ==========  ========  =========================
        """

        xi0 = self._get_xi0(nb_samples, n0)  # xi0[0] inferior, xi0[-1] superior
        # asym dyn shape
        # ---- --- ---------------
        #  F    F  (1,2) or (1,3)
        #  F    T  (n,2) or (n,3)
        #  T    F  (2,1,2) or (2,1,3)
        #  T    T  (2,n,2) or (2,n,3)

        # configure the prephonatory displacements (given in the order of inferior to superior)
        bulging = xi0.shape[-1] == 3
        if not bulging and self.Qb is not None:
            # mid-fold position not given, get bulging quotient if specified
            bulging = True
            Qb = _load_asym_param(self.Qb, 0, nb_samples, n0, force_data_axis=False)
            # asym dyn shape
            # ---- --- ---------------
            #  F    F  (1) or (1,1)
            #  F    T  (n,1) or (n,)
            #  T    F  (2,1)
            #  T    T  (2,n)

            Qb_has_lateral = Qb.ndim == 2 and Qb.shape[0] == 2

            delta_xi0 = np.abs((xi0[..., 1] - xi0[..., 0]) / 2)
            min_xi0 = xi0.min(axis=-1)
            xib = min_xi0 + (1 - Qb) * delta_xi0

            xi0_has_lateral = xi0.ndim == 3
            xib_has_lateral = xi0_has_lateral or Qb_has_lateral

            if xib.size == 1:
                xib = xib.reshape(())
            elif xib.ndim > 1 and xib.shape[0] == 1:
                xib = xib[0, ...]
            elif not xib_has_lateral and xi0_has_lateral:
                xib = xib[np.newaxis, ...]
            # elif not xi0_has_lateral and Qb_has_lateral:
            #     xi0 = xi0[np.newaxis, ...]

            bshape = np.broadcast_shapes(xi0.shape[:-1], xib.shape)
            xi0, xi0_ = np.empty((*bshape, 3)), xi0
            xi0[..., [0, 2]] = xi0_
            xi0[..., 1] = xib

        # define polynomial z-profile function
        xi0_shape = xi0.shape
        # if len(xi0_shape)==3 and xi0_shape[0] == 1:
        #     xi0_shape = xi0_shape[1:]
        while xi0_shape[0] == 1:
            xi0_shape = xi0_shape[1:]

        xi0 = xi0.reshape(-1, xi0.shape[-1])
        if bulging:
            # quadratic
            xi0_p = np.linalg.lstsq(
                [[1, 0, 0], [1, 0.5, 0.25], [1, 1, 1]], xi0.T, rcond=None
            )[0].T
        else:
            # linear
            xi0_p = np.linalg.lstsq([[1, 0], [1, 1]], xi0.T, rcond=None)[0].T

        z_shape = z.shape
        z = z.reshape(-1)
        zones = np.ones_like(z)
        zmat = np.stack([zones, z, z**2] if bulging else [zones, z], axis=0)
        return (xi0_p @ zmat).reshape((*xi0_shape[:-1], *z_shape))


class Xi0ZFunPoly(Xi0ZFunPolyBase):
    """Polynomial prephonatory posterior VF edge positioning function

    Signature of the returned function:

        xi0_zfun(z:NDArray, nb_samples:int, n0:int=0) -> NDArray

    If dynamic vocal folds (i.e., any of the pre-phonatory parameters is time dependent),
    output NDArray prepends the time-axes (shape: (nb_samples, *z.shape)) else matches the
    shape of z.

    This is the default implementation if user does not provide a callable xi0

    """

    xi0: SampleGenerator  # posterior edge prephonatory displacements in cm

    def __init__(
        self,
        xi0: ArrayLike | SampleGenerator,
        Qb: float | ArrayLike | SampleGenerator | None = None,
    ):
        """_summary_

        Parameters
        ----------
        xi0
            prephonatory posterior VF edge anchor positions. Specify either 2 or 3 point.
            For asymmetrical vocal folds, use 2x2 or 2x3 array. To make this time-varying,
            either use a 2-D SampleGenerator or 2-D or 3-D array with the first axis as
            the time axis. The time axis is sampled at the simulation sampling rate.
        Qb, optional
            bulging quotient if xi0 specifies only 2 endpoints, by default None

        Raises
        ------
        TypeError
            _description_
        """
        super().__init__(Qb)

        self.xi0: SampleGenerator = format_parameter(
            xi0, shape=[(2,), (3,), (2, 2), (2, 3)]
        )
        if self.xi0.shape[0] == 3 and Qb is not None:
            raise TypeError("Qb must be None when xi0 is a 3-element array.")

    def _get_xi0(self, nb_samples: int | None, n0: int = 0):
        """get prephonatory posterior VF edge at most inferior & superior depths

        Parameters
        ----------
        nb_samples
            simulation duration in number of time samples
        n0, optional
            simulation starting time, by default 0

        Returns
        -------
        xi0
            prephonatory posterior VF edge at the most inferior & superior depths. The shape
            of the array differs depending on the configuration:

            ==========  ========  ========================================
            symmetry    temporal  shape
            ==========  ========  ========================================
            symmetric   fixed     (2,) or (3,)
            symmetric   dynamic   (nb_samples, 2) or (nb_samples, 3)
            asymmetric  fixed     (2, 1, 2) or (2, 1, 3)
            asymmetric  dynamic   (2, nb_samples, 2) or (2, nb_samples, 3)
            ==========  ========  ========================================

        """

        return _load_asym_param(self.xi0, 1, nb_samples, n0)


class Xi0ZFunPolyRel(Xi0ZFunPolyBase):
    """prephonatory VF edge displacements relative to the maximum phonatory displacement

    Signature of the returned function:

    xi0_zfun(z:NDArray, nb_samples:int, n0:int=0) -> NDArray

    If dynamic vocal folds (i.e., any of the pre-phonatory parameters is time dependent),
    output NDArray prepends the time-axes (shape: (nb_samples, *z.shape)) else matches the
    shape of z.

    This is the default implementation if user does not provide a callable xi0

    """

    xim: SampleGenerator  # maximum displacement during phonation
    Qa: SampleGenerator = Constant(kinematic_params_default["Qa"])
    Qs: SampleGenerator = Constant(kinematic_params_default["Qs"])

    def __init__(
        self,
        xim: float | ArrayLike | SampleGenerator,
        Qa: float | ArrayLike | SampleGenerator | None = None,
        Qs: float | ArrayLike | SampleGenerator | None = None,
        Qb: float | ArrayLike | SampleGenerator | None = None,
    ):
        super().__init__(Qb)
        self.xim = format_parameter(xim, shape=[(), (1,), (2,)])
        if Qa is not None:
            self.Qa = format_parameter(Qa, shape=[(), (1,), (2,)])
        if Qs is not None:
            self.Qs = format_parameter(Qs, shape=[(), (1,), (2,)])

    def _get_xi0(self, nb_samples: int | None, n0: int = 0) -> NDArray:
        """calculate prephonatory posterior VF edge at most inferior & superior depths

        Parameters
        ----------
        nb_samples
            simulation duration in number of time samples
        n0, optional
            simulation starting time, by default 0

        Returns
        -------
        xi0
            prephonatory posterior VF edge at the most inferior & superior depths. The shape
            of the array differs depending on the configuration:

            ==========  ========  ==================
            symmetry    temporal  shape
            ==========  ========  ==================
            symmetric   fixed     (1, 2)
            symmetric   dynamic   (nb_samples, 2)
            asymmetric  fixed     (2, 1, 2)
            asymmetric  dynamic   (2, nb_samples, 2)
            ==========  ========  ==================

        """

        xim = _load_asym_param(self.xim, 0, nb_samples, n0)
        Qa = _load_asym_param(self.Qa, 0, nb_samples, n0)
        Qs = _load_asym_param(self.Qs, 0, nb_samples, n0)

        xi02 = xim * Qa
        xi01 = Qs * xim + xi02

        xi01_is_lateral = xi01.ndim > 1 and xi01.shape[0] > 1
        xi02_is_lateral = xi02.ndim > 1 and xi02.shape[0] > 1
        n = max(xi01.shape[-1], xi02.shape[-1])

        xi0 = np.empty((2, n, 2) if xi01_is_lateral or xi02_is_lateral else (n, 2))
        xi0[..., 0] = xi01
        xi0[..., 1] = xi02
        return xi0


Xi1Y_Callable = Callable[[NDArray, int, int], NDArray]
"""calculate the variations along the length (y-axis) of edge displacements 

        Parameters
        ----------
        y
            normalized y-axis, 0=most anterior, 1=most posterior
        nb_samples
            simulation duration in number of time samples
        n0
            simulation starting time

        Returns
        -------
        xi1
            Normalized superior VF edge displacement with respect 
            to prephonatory position along the length of vocal 
            fold specified by y. xi1(n,y) = 1.0 specifies that
            at the y along the length on the time index n, the 
            vocal fold opens to the full width (determined by
            the width)
            
            The shape of the array differs depending on the configuration:

            ==========  ========  =========================
            symmetry    temporal  shape
            ==========  ========  =========================
            symmetric   fixed     y.shape
            symmetric   dynamic   (nb_samples, *y.shape)
            asymmetric  fixed     (2, 1, *y.shape)
            asymmetric  dynamic   (2, nb_samples, *y.shape)
            ==========  ========  =========================

        Notes
        -----
        Left-right asymmetry in xi1 
"""


def xi1_yfun_mode1(y, n, n0):
    """mode-1 VF edge displacement variations along length (y-axis)

    Parameters
    ----------
    y
        normalized y-axis, 0=most anterior, 1=most posterior
    nb_samples
        simulation duration in number of time samples (not used)
    n0
        simulation starting time (not used)

    Returns
    -------
    xi1
        Time-invariant normalized variability of the superior VF edge
        displacement along the length of vocal fold specified by y
        with respect to prephonatory position and the reference motion.
        The shape of the output matches y.shape.
    """

    return np.sin(pi * y)


Xi1Z_Callable = Callable[[NDArray, int, int], NDArray]
"""calculate the variations along the depth (z-axis) of edge displacements 

        Parameters
        ----------
        z
            normalized z-axis, 0=most inferior, 1=most superior
        nb_samples
            simulation duration in number of time samples
        n0
            simulation starting time

        Returns
        -------
        xi1
            Normalized VF edge displacement variation with respect 
            to the reference displacement along the depth of vocal 
            fold specified by z. xi1 may be complex-valued to model
            the amplitude and phase variation.
            
            The shape of the array differs depending on the configuration:

            ==========  ========  =========================
            symmetry    temporal  shape
            ==========  ========  =========================
            symmetric   fixed     z.shape
            symmetric   dynamic   (nb_samples, *z.shape)
            asymmetric  fixed     (2, 1, *z.shape)
            asymmetric  dynamic   (2, nb_samples, *z.shape)
            ==========  ========  =========================
"""


class LeTalkerXi1ZFun(Xi1Z_Callable):
    Qnp: SampleGenerator = Constant(kinematic_params_default["Qnp"])
    Qp: SampleGenerator = Constant(kinematic_params_default["Qp"])

    def __init__(
        self,
        Qnp: float | ArrayLike | SampleGenerator | None = None,
        Qp: float | ArrayLike | SampleGenerator | None = None,
    ):
        """Abstract base of polynomial prephonatory posterior VF edge positioning functor

        Parameters
        ----------
        Qb, optional
            bulging quotient if xi0 specifies only 2 endpoints, by default None (no bulge)
        """
        if Qnp is not None:
            self.Qnp = format_parameter(Qnp, optional=True, shape=[(), (1,), (2,)])
        if Qp is not None:
            self.Qp = format_parameter(Qp, optional=True, shape=[(), (1,), (2,)])

    def __call__(self, z: NDArray, nb_samples: int | None, n0: int = 0) -> NDArray:
        """calculate the variations along the depth (z-axis) of edge displacements

        Parameters
        ----------
        z
            normalized z-axis, 0=most inferior, 1=most superior
        nb_samples
            simulation duration in number of time samples
        n0
            simulation starting time

        Returns
        -------
        xi1
            Normalized VF edge displacement variation with respect
            to the reference displacement along the depth of vocal
            fold specified by z. xi1 may be complex-valued to model
            the amplitude and phase variation.

            The shape of the array differs depending on the configuration:

            ==========  ========  =========================
            symmetry    temporal  shape
            ==========  ========  =========================
            symmetric   fixed     z.shape
            symmetric   dynamic   (nb_samples, *z.shape)
            asymmetric  fixed     (2, 1, *z.shape)
            asymmetric  dynamic   (2, nb_samples, *z.shape)
            ==========  ========  =========================

            The output is asymmetric if Qp or Qnp properties defines
            consists of 2 values.
        """

        Qp = _load_asym_param(self.Qp, 0, nb_samples, n0)
        Qnp = _load_asym_param(self.Qnp, 0, nb_samples, n0)

        if z.ndim > 1:
            Qp = Qp.reshape(*Qp.shape, *np.ones(z.ndim - 1, int))
            Qnp = Qnp.reshape(*Qnp.shape, *np.ones(z.ndim - 1, int))

        # amplsin*(sth - omega*(z-zn)*cth/c)
        return np.exp(-1j * _2pi * Qp * (z - Qnp))


###########################################


class KinematicVocalFolds(VocalFoldsAgBase):
    """Kinematic vocal fold model (Titze 1984)"""

    L0: float = constants.male_vf_params["L0"]
    T0: float = constants.male_vf_params["T0"]
    fo2L: Fo2L_Callable = staticmethod(
        partial(
            fo_to_L,
            **{k: v for k, v in zip(("A", "B"), constants.male_vf_params["fo2L"])},
        )
    )

    xi_ref: AnalyticFunctionGenerator
    xim: SampleGenerator = Constant(constants.male_vf_params["xim"])

    _xi0_zfun: Xi0Z_Callable

    _xi1_yfun: Xi1Y_Callable = staticmethod(xi1_yfun_mode1)
    _xi1_zfun: Xi1Z_Callable = LeTalkerXi1ZFun()

    _Ny: int = 21
    _Nz: int = 15

    @overload
    def __init__(
        self,
        fo_or_xi_ref: float | ArrayLike | AnalyticFunctionGenerator,
        L0: float | None = None,
        T0: float | None = None,
        xim: float | ArrayLike | SampleGenerator | None = None,
        fo2L: tuple[float, float] | Callable | None = None,
        *,
        Qa: float | ArrayLike | SampleGenerator | None = None,
        Qs: float | ArrayLike | SampleGenerator | None = None,
        Qb: float | ArrayLike | SampleGenerator | None = None,
        Qnp: float | ArrayLike | SampleGenerator | None = None,
        Qp: float | ArrayLike | SampleGenerator | None = None,
        xi1_yfun: Xi1Y_Callable | None = None,
        xi1_zfun: Xi1Z_Callable | None = None,
        upstream: VocalTract | float | None = None,
        downstream: VocalTract | float | None = None,
        aspiration_noise: bool | dict | AspirationNoise | None = None,
        Ny: int = 21,
        Nz: int = 15,
    ):
        """Kinematic vocal fold model (relative dimension)

        Parameters
        ----------
        fo_or_xi_ref
            analytic edge displacement periodic waveform or sinusoidal frequency
        L0
            relaxed vocal fold length in cm (1.6 for male, 1.0 for female)
        T0
            relaxed vocal fold thickness in cm (0.3 for male, 0.2 for female)
        fo2L, optional
            fo to vf length converter, by default None to use L0 and T0 as the vibrating length and
            thickness of the vocal fold
        xim, optional
            common amplitude of the upper and lower edges of the folds in cm, by default 0.1
        Qa, optional
            abduction quotient: xi02/xim (Titze 84 Eq (8)), by default kinematic_params_default['Qa']
        Qs, optional
            shape quotient: (xi01-xi02)/xim (Titze 84 Eq (9)), by default kinematic_params_default['Qs']
        Qb, optional
            bulging quotient: (|xi01-xi02|/2-(xib-xi02))/xim, by default kinematic_params_default['Qb']
        Qp, optional
            phase quotient: phi/2pi (Titze 84 Eq (10)), by default kinematic_params_default['Qp']
        Qnp, optional
            nodal point quotient, by default kinematic_params_default['Qnp']
        xi1_yfun, optional
            function returning complex-valued xi1 multipliers along y-axis, defaults to sin(pi*y/L),
            by default None
        xi1_zfun, optional
            function returning complex-valued xi1 muliplier along z-axis, defaults to LeTalker mechanism
            with Qp & Qnp.
        upstream, optional
            subglottal vocal tract object, by default None (open space)
        downstream, optional
            epiglottal vocal tract object area, by default None (open space)
        Ny, optional
            number of spatial simulation cells in y (AP)-axis, by default 21
        Nz, optional
            number of spatial simulation cells in z (transversal)-axis, by default 15



        """

    @overload
    def __init__(
        self,
        fo_or_xi_ref: float | ArrayLike | AnalyticFunctionGenerator,
        L0: float | None = None,
        T0: float | None = None,
        xim: float | ArrayLike | SampleGenerator | None = None,
        fo2L: tuple[float, float] | Callable | None = None,
        *,
        xi0: (
            tuple[float, float]
            | tuple[float, float, float]
            | ArrayLike
            | SampleGenerator
            | Callable
        ),
        Qb: float | ArrayLike | SampleGenerator | None = None,
        Qnp: float | ArrayLike | SampleGenerator | None = None,
        Qp: float | ArrayLike | SampleGenerator | None = None,
        xi1_yfun: Xi1Y_Callable | None = None,
        xi1_zfun: Xi1Z_Callable | None = None,
        upstream: VocalTract | None = None,
        downstream: VocalTract | None = None,
        aspiration_noise: bool | dict | AspirationNoise | None = None,
        Ny: int = 21,
        Nz: int = 15,
        **_,
    ):
        """Kinematic vocal fold model (absolute dimension)

        Parameters
        ----------
        fo_or_xi_ref
            analytic edge displacement periodic waveform or sinusoidal frequency
        L0
            relaxed vocal fold length in cm (1.6 for male, 1.0 for female)
        T0
            relaxed vocal fold thickness
        fo2L
            fo to vf length converter, by default None to use L0 and T0 as the vibrating length and
            thickness of the vocal fold
        xim, optional
            common amplitude of the upper and lower edges of the folds in cm, by default 0.1
        xi0, optional
            prephonatory displacements of the posterior edge of the vf
        Qp, optional
            phase quotient: phi/2pi (Titze 84 Eq (10)), by default kinematic_params_default['Qp']
        Qb, optional
            bulging quotient: (|xi01-xi02|/2-(xib-xi02))/xim, by default kinematic_params_default['Qb']
        Qnp, optional
            nodal point quotient, by default kinematic_params_default['Qnp']
        xi1_yfun, optional
            function returning xi0 multipliers along y-axis, defaults to sin(pi*y/L), by default None
        xi1_zfun, optional
            function returning complex-valued xi1 muliplier along z-axis, defaults to LeTalker mechanism
            with Qp & Qnp.
        upstream, optional
            subglottal vocal tract object, by default None (open space)
        downstream, optional
            epiglottal vocal tract object area, by default None (open space)
        Ny, optional
            number of spatial simulation cells in y (AP)-axis, by default 21
        Nz, optional
            number of spatial simulation cells in z (transversal)-axis, by default 15

        References
        ----------
        [1] I. R. Titze, “Parameterization of the glottal area, glottal flow, and vocal fold contact area,”
            J. Acoust. Soc. Am., vol. 75, no. 2, pp. 570–580, 1984, doi: 10.1121/1.390530.

        """

    def __init__(
        self,
        fo_or_xi_ref: float | ArrayLike | AnalyticFunctionGenerator,
        L0: float | None = None,
        T0: float | None = None,
        xim: float | ArrayLike | SampleGenerator | None = None,
        fo2L: tuple[float, float] | Fo2L_Callable | None = None,
        *,
        upstream: VocalTract | None = None,
        downstream: VocalTract | None = None,
        aspiration_noise: bool | dict | AspirationNoise | None = None,
        Ny: int = 21,
        Nz: int = 15,
        **kwargs,
    ):

        super().__init__(
            upstream=upstream, downstream=downstream, aspiration_noise=aspiration_noise
        )

        # set the reference motion function
        if isinstance(fo_or_xi_ref, AnalyticFunctionGenerator):
            self.xi_ref = fo_or_xi_ref
        elif isinstance(fo_or_xi_ref, FunctionGenerator):
            self.xi_ref = SineGenerator(fo_or_xi_ref)
        else:
            fo = np.asarray(fo_or_xi_ref, dtype=float)
            if (not fo.ndim) or (fo.ndim == 1 and len(fo) == 2):
                # symmetric fo
                self.xi_ref = SineGenerator(fo)
            elif not (fo.ndim == 1 or (fo.ndim == 2 and fo.shape[1] == 2)):
                raise TypeError(f"Invalid fo_or_xi_ref array shape: {fo.shape}")
            else:
                self.xi_ref = PeriodicInterpolator(self.fs, fo)

        if L0 is not None:
            self.L0 = float(L0)
        if T0 is not None:
            self.T0 = float(T0)
        if xim is not None:
            self.xim = format_parameter(xim, optional=True, shape=[(), (1,), (2,)])

        if fo2L is not None:
            if not callable(fo2L):
                try:
                    A, B = fo2L
                except TypeError as e:
                    # use the default male coefficients
                    raise TypeError(
                        "fo2L must be 2-float sequence to define A & B coefficients to run fo_to_L()"
                    ) from e
                fo2L = staticmethod(partial(fo_to_L, A=A, B=B))
            self.fo2L = fo2L

        if "Ny" in kwargs:
            self._Ny = int(Ny)
        if "Nz" in kwargs:
            self._Nz = int(Nz)

        # initialize the rest of parameters
        self._init_geometry(**kwargs)

    def _init_geometry(
        self,
        Qa: float | ArrayLike | SampleGenerator | None = None,
        Qs: float | ArrayLike | SampleGenerator | None = None,
        Qb: float | ArrayLike | SampleGenerator | None = None,
        xi0: ArrayLike | SampleGenerator | Xi0Z_Callable | None = None,
        Qnp: float | ArrayLike | SampleGenerator | None = None,
        Qp: float | ArrayLike | SampleGenerator | None = None,
        xi1_yfun: Xi1Y_Callable | None = None,
        xi1_zfun: Xi1Z_Callable | None = None,
    ):
        """set prephonatory geometry and spatial variation"""

        # select the prephonatory geometry along the depth
        self._xi0_zfun = (
            Xi0ZFunPolyRel(self.xim, Qa, Qs, Qb)
            if xi0 is None
            else (
                xi0
                if not isinstance(xi0, SampleGenerator) and isinstance(xi0, Callable)
                else Xi0ZFunPoly(xi0, Qb)
            )
        )

        # if custom y-axis vibration variation is given
        if xi1_yfun is not None:
            self._xi1_yfun = xi1_yfun

        # if custom z-axis vibration variation is given
        if xi1_zfun is not None:
            self._xi1_zfun = xi1_zfun
        elif Qnp is not None or Qp is not None:
            self._xi1_zfun = LeTalkerXi1ZFun(Qnp, Qp)

    def xi1_z(self, nb_samples: int, n0: int = 0) -> NDArray:
        z = np.linspace(0, 1, self._Nz)
        return self._xi1_zfun(z, nb_samples, n0)

    def xi1_y(self, nb_samples: int, n0: int = 0) -> NDArray:
        y = np.linspace(0, 1, self._Ny)
        return self._xi1_yfun(y, nb_samples, n0)

    def fo(self, nb_samples: int, n0: int = 0) -> NDArray:
        """get fundamental frequency

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            _description_
        """
        return self.xi_ref.fo(nb_samples, n0, force_time_axis=False)

    @property
    def known_length(self) -> bool:
        """True if model specifies the vibrating vocal fold length"""
        return True

    def length(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """vibrating glottal length

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            glottal length in cm. 1D array if fo or L0 is dynamic otherwise a scalar value.
        """

        fo = self.xi_ref.fo(nb_samples or 1, n0)
        if nb_samples is None:
            fo = np.squeeze(fo, 0)
        elif self.xi_ref.fo.ndim:
            fo = fo.mean(1)  # average left & right

        return self.fo2L(fo, self.L0)

    def thickness(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """glottal thickness

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            glottal thickness in cm. 1D array if fo or L0 is dynamic otherwise a scalar value.
        """

        length = self.length(nb_samples, n0)

        # thickness with length stretch factor [different from Titze-Story 2002 Eq (55)]
        epsilon = length / self.L0 - 1  # longitudinal vf strain
        return np.atleast_1d(self.T0 / (1 + 0.8 * epsilon))  # Titze-Story 2002 Eq (59)

    def xi0(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """prephonatory glottal geometry

        Parameters
        ----------
        nb_samples, optional
            number of samples, required if dynamic, by default None
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            2D array spanning y and z of prephonatory lateral displacements in cm.
            If dynamic, 3D array spanning time, y, and z.
        """

        # normalized y and z grid vectors

        # z: 0=inferior -> 1=superior
        z = np.linspace(0, 1, self._Nz, endpoint=True).reshape(1, -1)
        xi0_z = self._xi0_zfun(z, nb_samples, n0)

        # 2D if time-independent or 3D if dependent (+1 if asymmetric)

        # prephonatory y-axis magnitude shaping (V shape with vocal process distance set by xi0)
        # y: 0=posterior -> 1=anterior
        Y0 = np.linspace(1, 0, self._Ny, endpoint=True).reshape(-1, 1)

        # prephonatory glottal configuration
        return Y0 * xi0_z

    def xi(self, nb_samples: int, n0: int = 0) -> NDArray:
        """compute vocal fold edge displacement

        Parameters
        ----------
        nb_samples
            number of samples to generate
        n0, optional
            starting sample index, by default 0

        Returns
        -------
            VF edge position array where 0 is the glottal midline. If asymmetric,
            each vocal fold position is defined on a separate coordinate, each
            increasing away from the midline. The shape of the array is
            (nb_samples, ny, nz) for a symmetric model and (2, nb_samples, ny, nz)
            for asymmetric model.
        """

        ny = self._Ny
        nz = self._Nz

        # configure pre-phonatory geometry
        xi0 = self.xi0(nb_samples, n0)
        if xi0.ndim == 3 and xi0.shape[0] == 2:  # asymmetric stationary case
            xi0 = xi0[:, np.newaxis, ...]

        # get analytic reference motion (lateral, time, y, z)
        xi_ref = _load_asym_param(self.xi_ref, 0, nb_samples, n0, analytic=True)
        shape = xi_ref.shape
        if shape[0] == nb_samples:  # xi_ref symmetric
            xi_ref = xi_ref.reshape(*shape, *np.ones(3 - xi_ref.ndim, int))
        else:
            xi_ref = xi_ref.reshape(*shape, *np.ones(4 - xi_ref.ndim, int))

        # get maximum vibration amplitude
        xim = _load_asym_param(self.xim, 0, nb_samples, n0)
        shape = xim.shape
        if shape[0] == nb_samples:  # xi_ref symmetric
            xim = xim.reshape(*shape, *np.ones(3 - xim.ndim, int))
        else:
            xim = xim.reshape(*shape, *np.ones(4 - xim.ndim, int))

        # vibration magnitude shaping factor along y-axis
        y = np.linspace(0, 1, ny).reshape(-1, 1)
        Y1 = self._xi1_yfun(y, nb_samples, n0)

        # vibration phase delay in the z-axis
        z = np.linspace(0, 1, nz)
        Z1 = self._xi1_zfun(z, nb_samples, n0)

        # vocal fold displacements
        xi1 = (xi_ref * Z1 * Y1).real * xim

        # position
        xi = xi1 + xi0

        if xi.ndim < 4 or xi.shape[0] == 1:  # symmetric
            if xi.ndim == 4:
                # remove the singular lateral axis
                xi = xi[0]

            # negative == collision
            xi = np.maximum(0, xi)
        else:  # asymmetric
            tf = xi[0] < -xi[1]
            xi[0, tf] = (xi[0, tf] - xi[1, tf]) / 2
            xi[1, tf] = -xi[0, tf]

        return xi

    def glottal_widths(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """glottal widths (minimum width along the depth)

        :param nb_samples: number of time samples, defaults to None
        :param n0: starting sample time, defaults to 0
        :return: 2D array (time x ny) of glottal widths measures in cm0
        """

        xi = self.xi(nb_samples, n0)  # vf surface displacements from the midplane
        if xi.ndim < 4:  # symmetric
            w = 2 * xi
        else:  # asymmetric
            w = xi.sum(0)
        return w.min(axis=-1)  # take the narrowest width along the depth

    def glottal_area(self, nb_samples: int, n0: int = 0) -> NDArray:
        """glottal area (minimum opening along the depth)

        :param nb_samples: _description_
        :param n0: starting sample time, defaults to 0
        :return: glottal area samples
        """
        xi = self.glottal_widths(nb_samples, n0)  # 2D[time,y]

        # configure the length and thickness
        L = self.length(nb_samples, n0)

        # glottal area calculation (trapezoid x 2, left & right)
        g = xi[:, :-1] + xi[:, 1:]
        area = np.sum(g, axis=1) * (L / g.shape[1] / 2)  #

        return area

    @property
    def runner_info(self) -> tuple[type, list[tuple[str, type]]]:

        return KinematicVocalFolds.Runner, VocalFoldsAgBase.RunnerSpec

    def y(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        length = self.length(nb_samples, n0)
        if length.ndim and length.shape[0] > 1:
            length = length[:, np.newaxis]
        return length * np.linspace(0, 1, self._Ny)

    def z(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        thickness = self.thickness(nb_samples, n0)
        if thickness.ndim and thickness.shape[0] > 1:
            thickness = thickness[:, np.newaxis]
        return thickness * np.linspace(0, 1, self._Nz)

    def zindex_closest(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """z-index where prephonatory vocal fold is closest to the midline"""

        return self.xi0(nb_samples, n0)[..., 0, :].argmin(axis=-1)

    def contact_area(self, nb_samples: int | None = None, n0: int = 0) -> NDArray:
        """glottal contact area

        Parameters
        ----------
        nb_samples, optional
            _description_, by default None
        n0, optional
            _description_, by default 0

        Returns
        -------
            _description_
        """

        # simply add up the number of  elements that have crossed the midline if symmetric
        # or if vocal folds have the same

        xi = self.xi(nb_samples, n0)

        if xi.ndim < 4:
            tf = (xi == 0).astype(int)
        else:
            tf = (xi[0] == -xi[1]).astype(int)

        xminus = np.sum(
            tf[:, :-1, :-1] + tf[:, 1:, :-1] + tf[:, :-1, 1:] + tf[:, 1:, 1:],
            axis=(1, 2),
        )

        dy = self.length(nb_samples, n0) / (tf.shape[1] - 1)
        dz = self.thickness(nb_samples, n0) / (tf.shape[2] - 1)

        return dy * dz / 4 * xminus

    def draw3d(
        self,
        n: int,
        *,
        axes: "Axes3D" | None = None,
        update: bool | None = None,
        units: Literal["cm", "mm"] = "cm",
        xmax: float | None = None,
        right_only: bool = False,
        draw_box: bool = False,
        obj_kws: dict[Literal["rfold", "lfold", "rbox", "lbox"], dict] | None = None,
    ):
        """draw the 3D vocal folds at specified time

        Parameters
        ----------
        n
            time sample index
        axes, optional
            matplotlib 3D axes to plot on, by default None
        update, optional
            True to update existing drawing on the axes, by default None (auto-detect)
        units, optional
            specify the spatial units, by default "cm"
        xmax, optional
            extent of the x-axis, spanning -xmax to xmax by default None
        right_only, optional
            True to only show right vocal folds, by default False
        draw_box, optional
            True to draw box to represent larynx, by default False
        obj_kws, optional
            Matplotlib 3D object attributes to customize the look of
            the 3D parts, by default None.
        """

        # get data

        xi = self.xi(1, n0=n)  # returns 2D or 3D
        is_symmetric = xi.ndim == 3

        y = self.y(n0=n)
        z = self.z(n0=n)

        if y.ndim > 1:
            y = y[np.argmax(y[:, -1])]
        if z.ndim > 1:
            z = z[np.argmax(z[:, -1])]

        if units == "mm":
            xi *= 10
            y *= 10
            z *= 10

        if xmax is None:
            xmax = np.max(xi) * 1.2

        Xr = xi[0] if is_symmetric else xi[0, 0]
        Xl = -xi[0] if is_symmetric else -xi[1, 0]

        Z, Y = np.meshgrid(z, y)

        gid0 = "letalker.kvf"

        updated = False

        if update or update is None:
            from matplotlib import cbook

            for h in axes.findobj(
                match=lambda artist: (artist.get_gid() or "").startswith(gid0),
                include_self=False,
            ):
                updated = True

                name = h.get_gid().rsplit(".", 1)[1]
                if name.endswith("fold"):
                    h.set_verts(
                        np.stack(
                            [
                                cbook._array_patch_perimeters(a, 1, 1)
                                for a in (Xr if name[0] == "r" else Xl, Y, Z)
                            ],
                            axis=-1,
                        )
                    )
                else:
                    if name.endswith("topface"):
                        X = Xl if name[0] == "l" else Xr
                        xy = np.array(
                            [[*X[:, -1], xmax, xmax, X[0, -1]], [*y, y[-1], y[0], y[0]]]
                        ).T
                        zs = z[-1]
                        zdir = "z"
                    elif name.endswith("bottomface"):
                        X = Xl if name[0] == "l" else Xr
                        xy = np.array(
                            [[*X[:, 0], xmax, xmax, X[0, 0]], [*y, y[-1], y[0], y[0]]]
                        ).T
                        zs = z[0]
                        zdir = "z"
                    elif name.endswith("frontface"):
                        X = Xl if name[0] == "l" else Xr
                        xy = np.array(
                            [[*X[0, :], xmax, xmax, X[0, 0]], [*z, z[-1], z[0], z[0]]]
                        ).T
                        zs = y[0]
                        zdir = "y"
                    elif name.endswith("backface"):
                        X = Xl if name[0] == "l" else Xr
                        xy = np.array(
                            [[*X[-1, :], xmax, xmax, X[-1, 0]], [*z, z[-1], z[0], z[0]]]
                        ).T
                        zs = y[-1]
                        zdir = "y"
                    elif name.endswith("endface"):
                        xy = np.array([y[[0, -1, -1, 0, 0]], z[[0, 0, -1, -1, 0]]]).T
                        zs = xmax
                        zdir = "x"

                    if name[0] == "l":
                        if zdir == "x":
                            zs = -zs
                        else:
                            xy[:, 0] = -xy[:, 0]
                    h.set_3d_properties(xy, zs, zdir)

        if update is False or (update is None and not updated):
            from matplotlib.patches import Polygon
            from mpl_toolkits.mplot3d.art3d import patch_2d_to_3d

            # gather custom properties
            obj_kws = obj_kws or {}
            for s in "lr":
                fname = f"{s}fold"
                obj_kws[fname] = {**obj_kws.get("folds", {}), **obj_kws.get(fname, {})}
                gname = f"{s}faces"
                for f in (
                    "top",
                    "bottom",
                    "front",
                    "back",
                    "end",
                ):
                    fname = f"{s}{f}face"
                    obj_kws[fname] = {
                        **obj_kws.get("faces", {}),
                        **obj_kws.get(gname, {}),
                        **obj_kws.get(fname, {}),
                    }

            # plot the data
            axes.plot_surface(
                Xr, Y, Z, gid=f"{gid0}.rfold", **{"edgecolors": "k", **obj_kws["rfold"]}
            )

            if not right_only:
                axes.plot_surface(
                    Xl,
                    Y,
                    Z,
                    gid=f"{gid0}.lfold",
                    **{"edgecolors": "k", **obj_kws["lfold"]},
                )

            def plot_face(xy, name, zs, zdir, plot_left=False):
                if plot_left:
                    p = Polygon(
                        xy,
                        closed=False,
                        gid=f"{gid0}.r{name}",
                        **{"fc": "none", "ec": "k", **obj_kws.get(f"r{name}")},
                    )
                    patch_2d_to_3d(p, zs, zdir)
                    axes.add_patch(p)
                elif not right_only:
                    # if zdir == "x":
                    #     zs = -zs
                    # else:
                    #     xy[:, 0] = -xy[:, 0]
                    p = Polygon(
                        xy,
                        closed=False,
                        gid=f"{gid0}.l{name}",
                        **{"fc": "none", "ec": "k", **obj_kws.get(f"l{name}")},
                    )
                    patch_2d_to_3d(p, zs, zdir)
                    axes.add_patch(p)

            if draw_box:
                # box faces
                for l, X in zip("lr", (Xl, Xr)):
                    plot_face(
                        np.array([[*X[:, -1], xmax, xmax], [*y, y[-1], y[0]]]).T,
                        "topface",
                        z[-1],
                        "z",
                    )
                    plot_face(
                        np.array([[*X[:, 0], xmax, xmax], [*y, y[-1], y[0]]]).T,
                        "bottomface",
                        z[0],
                        "z",
                    )
                    plot_face(
                        np.array([[*X[0, :], xmax, xmax], [*z, z[-1], z[0]]]).T,
                        "frontface",
                        y[0],
                        "y",
                    )
                    plot_face(
                        np.array([[*X[-1, :], xmax, xmax], [*z, z[-1], z[0]]]).T,
                        "backface",
                        y[-1],
                        "y",
                    )
                    plot_face(
                        np.array([y[[0, -1, -1, 0]], z[[0, 0, -1, -1]]]).T,
                        "endface",
                        xmax,
                        "x",
                    )

    @property
    def _runner_fields_to_results(self) -> list[str]:
        """list of runner fields to store in results"""
        return ["n", "ug", "psg", "peplx"]

    @dataclass
    class Results(Element.Results):
        """Simulation Results"""

        ug: NDArray
        psg: NDArray
        peplx: NDArray
        aspiration_noise: Element.Results | None

        @property
        def subglottal_pressure(self):
            return self.psg

        @cached_property
        def length(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.length(self.nb_samples, self.n0)

        @cached_property
        def y(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.y(self.nb_samples, self.n0)

        @cached_property
        def thickness(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            thickness = element.thickness(self.nb_samples, self.n0)
            return thickness

        @cached_property
        def z(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.z(self.nb_samples, self.n0)

        @cached_property
        def xi_ref(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.xi_ref(self.nb_samples, self.n0)

        @cached_property
        def xi0(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.xi0(self.nb_samples, self.n0)

        @cached_property
        def xi(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.xi(self.nb_samples, self.n0)

        @cached_property
        def displacements(self) -> NDArray:
            """maximum lateral displacements at the minimum opening along the depth"""
            return self.xi.min(-1).max(-1)

        @cached_property
        def fo(self) -> NDArray:
            """fundamental frequency of the referenc VF motion in Hz"""
            element = cast(KinematicVocalFolds, self.element)
            return element.fo(self.nb_samples, self.n0)

        @cached_property
        def phi(self) -> FunctionGenerator:
            """phase of the referenc VF motion in radians"""
            element = cast(KinematicVocalFolds, self.element)
            return element.xi_ref.phase(self.nb_samples, self.n0)

        @cached_property
        def zindex_closest(self) -> float:
            """z-index where prephonatory vocal fold is closest to the midline"""

            return self.xi0(self.nb_samples, self.n0)[..., 0, :].argmin(axis=-1)

        @cached_property
        def glottal_widths(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.glottal_widths(self.nb_samples, self.n0)

        @cached_property
        def glottal_area(self) -> NDArray:
            element = cast(KinematicVocalFolds, self.element)
            return element.glottal_area(self.nb_samples, self.n0)

        @cached_property
        def contact_area(self):
            # Contact area
            # simply add up the number of  elements that have crossed the midline
            element = cast(KinematicVocalFolds, self.element)
            return element.contact_area(self.nb_samples, self.n0)

        @cached_property
        def reynolds_number(self):
            return self.ug / (self.element.nu * self.length)

        @overload
        def draw3d(
            self,
            n: int,
            *,
            axes: "Axes3D" | None = None,
            update: bool | None = None,
            units: Literal["cm", "mm"] = "cm",
            xmax: float | None = None,
            right_only: bool = False,
            draw_box: bool = False,
            obj_kws: (
                dict[Literal["rfold", "lfold", "rbox", "lbox"], dict] | None
            ) = None,
        ):
            """draw the 3D vocal folds at specified time

            Parameters
            ----------
            n
                time index, relative to n0 of the simulation
            axes, optional
                matplotlib 3D axes to plot on, by default None
            update, optional
                True to update existing drawing on the axes, by default None (auto-detect)
            units, optional
                specify the spatial units, by default "cm"
            xmax, optional
                extent of the x-axis, spanning -xmax to xmax by default None
            right_only, optional
                True to only show right vocal folds, by default False
            draw_box, optional
                True to draw box to represent larynx, by default False
            obj_kws, optional
                Matplotlib 3D object attributes to customize the look of
                the 3D parts, by default None.
            """

        def draw3d(self, n: int, **kwargs):
            element = cast(KinematicVocalFolds, self.element)
            if n < 0 or n >= self.nb_samples:
                raise ValueError(
                    f"{n=} is out of simulated range [0, {self.nb_samples})."
                )
            return element.draw3d(n, **kwargs)

    # override result class
    _ResultsClass = Results
