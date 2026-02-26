from __future__ import annotations

from typing import Any
from numpy.typing import ArrayLike, NDArray

import numpy as np

from .core import has_numba, get_sampling_rate
from .function_generators.abc import SampleGenerator
from .function_generators import ClampedInterpolator, Constant


def parameter_is_dynamic(parameter: Any) -> bool:
    """Returns True if parameter is a function generator object"""
    return isinstance(parameter, SampleGenerator)


def format_parameter(
    parameter: float | ArrayLike | SampleGenerator | None,
    ndim: int = 0,
    shape: int | tuple[int, ...] | list[int | tuple[int, ...]] | None = None,
    is_static: bool = False,
    force_time_axis: bool = False,
    optional: bool = False,
) -> float | NDArray | SampleGenerator | None:
    """validate and format a element parameter

    Parameters    ----------
    parameter
        user supplied parameter value
    ndim
        Validates the number of dimensions (excluding the time axes if present), by default 0 (scalar parameter)
        If shape is specified, ndim will be ignored.
    shape
        Validates the parameter shape (excluding time axes if present), by default None (arbitrary shape)
    is_static
        If True parameter must be a static parameter, by default True
    force_time_axis
        If True a scalar parameter

    Returns
    -------
        Formatted parameter variable
            - float if a stationary scalar parameter
            - NDArray if a stationary array parameter
            - SampleGenerator if a dynamic parameter. If time samples are given in `parameter` argument,
              it uses a `ClampedInterpolator` `SampleGenerator` object to hold the edge values outside of its time range.
            - None if parameter is None and optional = True

    """

    if parameter is None:
        if optional:
            return None
        raise ValueError("Parameter cannot be None")

    if shape is not None:
        ndim = None

    if isinstance(shape, int):
        shape = [(shape,)]
    elif isinstance(shape, list):
        shape = [(s,) if isinstance(s, int) else s for s in shape]
    elif shape is not None:
        shape = [shape]

    def validate_shape(s: tuple[int, ...], s0: tuple[int, ...]) -> bool:
        if len(s) != len(s0):
            return False
        return all(si == s0i if si >= 0 else True for si, s0i in zip(s, s0))

    if isinstance(parameter, SampleGenerator):
        is_fixed = parameter.is_fixed
        if is_static and not is_fixed:
            raise ValueError(f"Parameter cannot be a dynamic SampleGenerator.")
        if (
            ndim
            and parameter.ndim != ndim
            or shape
            and (not any(validate_shape(parameter.shape, s) for s in shape))
        ):
            raise ValueError(f"Invalid parameter dimension/shape.")
    else:
        parameter = np.asarray(parameter)
        # assume stationary first
        pndim = parameter.ndim
        pshape = parameter.shape
        if ndim == pndim or (shape and any(validate_shape(pshape, s) for s in shape)):
            return Constant([parameter] if force_time_axis and not ndim else parameter)
        if is_static:
            raise ValueError(f"Invalid parameter dimension/shape.")

        # check dynamic array
        pndim -= 1
        pshape = pshape[1:]
        if (ndim and ndim == pndim) or (
            shape and any(validate_shape(pshape, s) for s in shape)
        ):
            raise ValueError(f"Invalid parameter dimension/shape.")

        parameter = ClampedInterpolator(
            get_sampling_rate(),
            [parameter] if force_time_axis and not ndim else parameter,
        )

    return parameter


def parameter_to_array(
    parameter: float | NDArray | SampleGenerator | None,
    nb_samples: int | None,
    n0: int = 0,
    ndim: int = 0,
) -> NDArray | None:
    """convert multi-type parameter to numpy array

    Parameters
    ----------
    parameter
        multi-type parameter, including time-dependent SampleGenerator type
    nb_samples
        number of parameter samples to generate if time dependent. Otherwise,
        prepends an time axes if nb_samples is not None.
    n0, optional
        starting sample index, by default 0
    ndim, optional
        number of dimensions of the parameter, by default 0 or scaler

    Returns
    -------
        _description_
    """
    if parameter is None:
        return None

    if parameter_is_dynamic(parameter):
        return parameter(nb_samples, n0)

    out = np.ascontiguousarray(parameter)
    if out.ndim == ndim:
        # constant value, prepend the time-axis
        out = out[np.newaxis, ...]
    return out
