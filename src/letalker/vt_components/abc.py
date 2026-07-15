from __future__ import annotations

import abc
from typing import Any, Protocol, Self, Sequence

import control as ct
import numpy as np

from ..core import TimeSampleHandler

__all__ = ["TwoPortSystem"]


class LTISegmentFactory(Protocol):
    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.LTI:
        """Generate an LTI model for a vocal tract segment

        Parameters
        ----------
        area
            cross-sectional area in cm²
        length
            segment length in cm
        fs, optional
            sampling rate in samples/second to create a discrete-time model,
            by default the created model will be a continuous-time model.
        sample_kws, optional
            Keyword arguments to run ``ct.sample()`` function to discretize
            the model, by default the default parameters of ``ct.sample()`` will
            be used.
        sample_last, optional
            True to generate the subsystems in the continuous-time then
            convert to the discrete time at the end, by default False to sample
            the original continous-time systems

        Returns
        -------
            Generated model
        """


class LTISourceFactory(Protocol):
    @property
    def ninputs(self) -> int:
        """number of input signals to drive the source"""

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.LTI:
        """Generate an LTI model with ninputs+1 (Backward port + source signal)
            inputs and 1 output (Forward port)

        Parameters
        ----------
        area
            cross-sectional area in cm²
        length
            segment length in cm
        fs, optional
            sampling rate in samples/second to create a discrete-time model,
            by default the created model will be a continuous-time model.
        sample_kws, optional
            Keyword arguments to run ``ct.sample()`` function to discretize
            the model, by default the default parameters of ``ct.sample()`` will
            be used.

        Returns
        -------
            Generated model
        """


class LTISinkFactory(Protocol):
    """acoustic radiation model factory"""

    @property
    def noutputs(self) -> int:
        """number of output signals to the sink produces"""

    def __call__(
        self,
        area: float,
        length: float,
        *,
        fs: float | None = None,
        sample_kws: dict[str, Any] | None = None,
        sample_last: bool = False,
    ) -> ct.LTI:
        """Generate an LTI model with 1 input (Forward port)
            and 1+noutputs output (Backwards port + radiation outputs)

        Parameters
        ----------
        area
            cross-sectional area in cm²
        length
            segment length in cm
        fs, optional
            sampling rate in samples/second to create a discrete-time model,
            by default the created model will be a continuous-time model.
        sample_kws, optional
            Keyword arguments to run ``ct.sample()`` function to discretize
            the model, by default the default parameters of ``ct.sample()`` will
            be used.

        Returns
        -------
            Generated model
        """


class TwoPortSystem(TimeSampleHandler, metaclass=abc.ABCMeta):
    @property
    def nb_elements(self) -> int:
        """Number of network elements"""
        return 1

    @property
    @abc.abstractmethod
    def nb_states(self) -> int:
        """Total number of internal states"""

    @property
    def nb_input_ports(self) -> int:
        """Number of input ports"""
        return 1

    @property
    def nb_aux_inputs(self) -> int:
        """Number of auxiliary inputs"""
        return 0

    @property
    def nb_inputs(self) -> int:
        """Total number of inputs"""
        return self.nb_input_ports * 2 + self.nb_aux_inputs

    @property
    def nb_outputs(self) -> int:
        """Number of outputs"""
        return 2

    @property
    def nb_output_ports(self) -> int:
        """Number of output ports"""
        return 1

    @abc.abstractmethod
    def ss(
        self, *, fs: float | None = None, sample_kws: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """generate state-space matrices of all sections as a sample generator

        Parameters
        ----------
        fs, optional
            Sampling rate in samples/second. If specified, the
        sample_kws, optional
            Specify the continuous-to-discrete time conversion options. See below
            for a brief summary of the available options or see Control Systems
            Library documentation.

        Returns
        -------
        A
            State matrix, 3D array of shape ``(nb_samples,nb_states,nb_states)``.
        B
            Input matrix, 3D array of shape ``(nb_samples,nb_states,nb_inputs)``.
            If auxilirary pressure source is enabled, the 3rd input is always pressure source,
            and the last input is always auxilirary flow source if enabled.
        C
            Output matrix, 3D array of shape ``(nb_samples,2,nb_states)``
        D
            Feed-through matrix, 3D array of shape ``(nb_samples,2,nb_inputs)``


        Sample Options
        --------------

        ``'method'``: Method to use for sampling:

            * ``'gbt'``: generalized bilinear transformation
            * ``'backward_diff'``: Backwards difference (``'gbt'`` with ``alpha=1.0``)
            * ``'bilinear'`` (or ``'tustin'``): Tustin's approximation (``'gbt'`` with ``alpha=0.5``) (default)
            * ``'euler'``: Euler (or forward difference) method (``'gbt'`` with ``alpha=0``)
            * ``'zoh'``: zero-order hold

        ``'alpha'``: ``float`` within [0, 1]
            The generalized bilinear transformation weighting parameter,
            which should only be specified with ``method='gbt'``, and is
            ignored otherwise.

        ``'prewarp_frequency'``: ``float`` within [0, infinity)
            The frequency [rad/s] at which to match with the input continuous-time
            system's magnitude and phase (the gain = 1 crossover frequency, for
            example). Should only be specified with ``method = 'bilinear'`` or
            ``'gbt'`` with ``alpha = 0.5`` and ignored otherwise.

        """

    def __add__(self, next: Self) -> Self:
        return self.cascade(next)

    def __rshift__(self, other: ct.LTI | dict[int, ct.LTI] | Sequence[ct.LTI]) -> Self:
        return self.terminate(tail=other)

    def __rrshift__(self, other) -> Self:
        return self.terminate(head=other)

    def has_head(self) -> bool:
        """True if the input port have been connected"""

    def has_tails(self) -> list[bool]:
        """Return a list of bool to indicate if the nth output port have been connected"""

    def cascade(
        self, next: Self, *, out_port: int = 0, in_port: int = 0
    ) -> "CascadeNetwork":
        """cascade two network elements into one

        Parameters
        ----------
        next
            next network element
        out_port, optional
            output port index of this element to connect from, by default 0
        in_port, optional
            input port index of the next element to connect to, by default 0

        Returns
        -------
            Cascaded network of self->next
        """

        from .cascade import CascadeNetwork

        return CascadeNetwork()

    def terminate(
        self,
        *,
        head: ct.LTI | None = None,
        tail: ct.LTI | dict[int, ct.LTI] | Sequence[ct.LTI] | None = None,
        head_port: int | None = None,
        tail_port: int | None = None,
    ) -> "CascadeNetwork":

        from .cascade import CascadeNetwork

        return CascadeNetwork()
