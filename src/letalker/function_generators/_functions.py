from __future__ import annotations

from typing import Sequence
from numpy.typing import NDArray

import builtins

import numpy as np
import numpy.polynomial.polynomial as npp
from math import prod

from .abc import FunctionGenerator

_2pi = 2 * np.pi

#######

### polynomial function


def poly(c: NDArray, t: NDArray) -> NDArray:
    x = npp.polyval(t, c)
    return np.moveaxis(x, -1, 0) if x.ndim else x


def poly_derivative(c: NDArray, t: NDArray, nu: int = 1) -> NDArray | float:
    x = npp.polyval(t, npp.polyder(c, nu))
    return np.moveaxis(x, -1, 0) if x.ndim else x


def poly_antiderivative(c: NDArray, t: NDArray, nu: int = 1) -> NDArray:
    x = npp.polyval(t, npp.polyint(c, nu))
    return np.moveaxis(x, -1, 0) if x.ndim else x


setattr(poly, "derivative", poly_derivative)
setattr(poly, "antiderivative", poly_antiderivative)


#######

### piecewise polynomial function


def pwpoly(c: Sequence[NDArray], tb: Sequence[float], t: NDArray) -> NDArray:

    shape = np.asarray(c[0]).shape[1:]  # function sample shape
    idx = (slice(None),) * len(shape)

    x = np.empty((*shape, len(t)))

    tf0 = t < tb[0]
    x[tuple((*idx, tf0))] = npp.polyval(t[tf0], c[0])
    tf0[:] = ~tf0
    for ci, t1 in zip(c[1:], tb[1:]):
        tf1 = t < t1
        tf = tf0 & tf1
        x[tuple((*idx, tf))] = npp.polyval(t[tf], ci)
        tf0 = ~tf1

    x[tuple((*idx, tf0))] = npp.polyval(t[tf0], c[-1])

    return np.moveaxis(x, -1, 0)


def pwpoly_derivative(
    c: Sequence[NDArray], tb: Sequence[float], t: NDArray, nu: int = 1
) -> NDArray | float:

    return pwpoly([npp.polyder(ci, nu) for ci in c], tb, t)


def pwpoly_antiderivative(
    c: Sequence[NDArray], tb: Sequence[float], t: NDArray, nu: int = 1
) -> NDArray:

    # set boundary conditions
    c = [ci for ci in c]
    for _ in range(1, nu + 1):
        c[0] = npp.polyint(c[0])
        for i, ti in enumerate(tb):
            x0 = npp.polyval(ti, c[i])
            c1 = npp.polyint(c[i + 1])
            x1 = npp.polyval(ti, c1)
            c1[0] = x0 - x1
            c[i + 1] = c1

    return pwpoly(c, tb, t)


setattr(pwpoly, "derivative", pwpoly_derivative)
setattr(pwpoly, "antiderivative", pwpoly_antiderivative)

##########


def sine(omega: float, phi0: float, t: NDArray, analytic: bool) -> NDArray:

    phi = omega * t + phi0
    return np.exp(1j * (phi - np.pi / 2)) if analytic else np.sin(phi)


def sine_derivative(
    omega: float, phi0: float, t: NDArray, analytic: bool, nu: int = 1
) -> NDArray:

    theta = omega * t + phi0

    s = (1j) ** nu
    if analytic:
        x = np.exp(1j * (theta - np.pi / 2))
    else:
        sin_or_cos = np.sin if np.isreal(s) else np.cos
        s = s.real if np.isreal(s) else s.imag
        x = sin_or_cos(theta)
    return s * x * omega**nu


def sine_antiderivative(
    omega: float, phi0: float, t: NDArray, analytic: bool, nu: int = 1
) -> NDArray:
    theta = omega * t + phi0
    s = (-1j) ** nu
    if analytic:
        x = np.exp(1j * (theta - np.pi / 2))
    else:
        sin_or_cos = np.sin if np.isreal(s) else np.cos
        s = s.real if np.isreal(s) else s.imag
        x = sin_or_cos(theta)
    return s * x * omega ** (-nu)


setattr(sine, "derivative", sine_derivative)
setattr(sine, "antiderivative", sine_antiderivative)

##########


def flutter(
    a: NDArray, omega: NDArray, phi0: NDArray, t: NDArray, analytic: bool
) -> NDArray:
    """sum of sinusoids, quasi-random continuous function"""

    phi = omega[:, np.newaxis] * t[np.newaxis, ...] + phi0[:, np.newaxis]
    return np.dot(a, np.exp(1j * (phi - np.pi / 2)) if analytic else np.sin(phi))


def flutter_derivative(
    a: NDArray, omega: NDArray, phi0: NDArray, t: NDArray, analytic: bool, nu: int = 1
) -> NDArray:

    omega = omega[:, np.newaxis]
    theta = omega * t[np.newaxis, ...] + phi0[:, np.newaxis]

    s = (1j) ** nu
    if analytic:
        x = np.exp(1j * (theta - np.pi / 2))
    else:
        sin_or_cos = np.sin if np.isreal(s) else np.cos
        s = s.real if np.isreal(s) else s.imag
        x = sin_or_cos(theta)
    return np.dot(s * a, x * omega**nu)


def flutter_antiderivative(
    a: NDArray, omega: NDArray, phi0: NDArray, t: NDArray, analytic: bool, nu: int = 1
) -> NDArray:
    omega = omega[:, np.newaxis]
    theta = omega * t[np.newaxis, ...] + phi0[:, np.newaxis]
    s = (-1j) ** nu
    if analytic:
        x = np.exp(1j * (theta - np.pi / 2))
    else:
        # s *= -1j
        sin_or_cos = np.sin if np.isreal(s) else np.cos
        s = s.real if np.isreal(s) else s.imag
        x = sin_or_cos(theta)
    return np.dot(a * s, x * omega ** (-nu))


setattr(flutter, "derivative", flutter_derivative)
setattr(flutter, "antiderivative", flutter_antiderivative)

########


def product(
    generators: Sequence[FunctionGenerator],
    n: int,
    n0: int = 0,
    sample_edges: bool = False,
    upsample_factor: int = 0,
    *,
    gain: float | NDArray | None = None,
    **kwargs,
) -> NDArray:
    """generate waveform by multiplying the outputs of multiple generators

    Parameters
    ----------
    n
        number of samples to generate

    n0
        initial sample index, defaults to 0

    """

    ref = max(generators, key=lambda g: g.ndim)
    ndim = ref.ndim

    return prod(
        (
            g(n, n0, sample_edges, upsample_factor, **kwargs).reshape(
                (-1, *np.ones(ndim - g.ndim, int), *g.shape)
            )
            for g in generators
        ),
        start=1 if gain is None else gain,
    )


def product_derivative(
    generators: Sequence[FunctionGenerator],
    n: int,
    n0: int = 0,
    sample_edges: bool = False,
    upsample_factor: int = 0,
    nu: int = 1,
    *,
    gain: float | NDArray | None = None,
    **kwargs,
) -> int:

    if nu > 1:
        raise NotImplementedError(
            "higher-order derivatives are not currently implemented"
        )

    kwargs["nu"] = nu

    ref = max(generators, key=lambda g: g.ndim, default=None)
    ndim = ref.ndim
    shape = ref.shape

    args = (n, n0, sample_edges, upsample_factor)

    nb_gens = len(generators)

    if nb_gens == 1:
        dx = generators[0].derivative(*args, **kwargs)
    else:
        # chain rule
        f = np.empty((nb_gens, n, *shape))
        df = np.empty_like(f)
        for i, g in enumerate(generators):
            new_shape = (-1, *np.ones(ndim - g.ndim, int), *g.shape)
            f[i] = g(*args, **kwargs).reshape(new_shape)
            df[i] = g.derivative(*args, **kwargs).reshape(new_shape)

        ff = np.cumprod(f[:-1], axis=0)
        fb = np.cumprod(f[-1:0:-1], axis=0)

        dx = df[0] * fb[-1] + df[-1] * ff[-1]

        for i, dfi in enumerate(df[1:-1]):
            dx += dfi * ff[i] * fb[nb_gens - i - 3]

    if gain is not None and np.any(gain != 1.0):
        dx *= gain

    return dx


def trapezoid_rule(f: NDArray, fs: float, P: int = 1) -> NDArray:
    """trapezoid rule to approximate integration

    Parameters
    ----------
    f
        n-sample signal array to be numerically integrated. If multi-dimensional
        the first dimension is assumed to be the time-axis. The samples are
        sampled at the (P*fs) samples/second.
    fs
        output sampling rate
    P, optional
        downsampling factor, by default 1

    Returns
    -------
        (n-1)//P-sample first antiderivative array of the input signal. The jth
        output sample is located at input time index of (j+1/2)*P.

    """

    return np.cumsum((f[:-1] + f[1:]).reshape((-1, P, *f.shape[1:])).sum(1), axis=0) / (
        2 * P * fs
    )


def product_antiderivative(
    generators: Sequence[FunctionGenerator],
    n: int,
    n0: int = 0,
    sample_edges: bool = False,
    upsample_factor: int = 0,
    nu: int = 1,
    *,
    gain: float | NDArray | None = None,
    **kwargs,
) -> int:

    if nu > 1:
        raise NotImplementedError(
            "higher-order derivatives are not currently implemented"
        )

    upsample_factor += 1

    # numerical integration trapazoid approximation at 2x rate
    f = product(generators, n, n0, True, upsample_factor, **kwargs)
    x = trapezoid_rule(f, generators[0].fs, 2)
    if gain is not None:
        x *= gain
    return x


setattr(product, "derivative", product_derivative)
setattr(product, "antiderivative", product_antiderivative)

########


def sum(
    generators: Sequence[FunctionGenerator],
    n: int,
    n0: int = 0,
    sample_edges: bool = False,
    upsample_factor: int = 0,
    *,
    bias: float | NDArray | None = None,
    **kwargs,
) -> NDArray:
    """generate waveform by multiplying the outputs of multiple generators

    Parameters
    ----------
    n
        number of samples to generate

    n0
        initial sample index, defaults to 0

    """

    ref = max(generators, key=lambda g: g.ndim)
    ndim = ref.ndim

    x = builtins.sum(
        (
            g(n, n0, sample_edges, upsample_factor, **kwargs).reshape(
                (-1, *np.ones(ndim - g.ndim, int), *g.shape)
            )
            for g in generators
        ),
        start=0.0 if bias is None else bias,
    )

    return x


def sum_derivative(
    generators: Sequence[FunctionGenerator],
    n: int,
    n0: int = 0,
    sample_edges: bool = False,
    upsample_factor: int = 0,
    nu: int = 1,
    **kwargs,
) -> int:

    ref = max(generators, key=lambda g: g.ndim)
    ndim = ref.ndim

    return builtins.sum(
        g.derivative(n, n0, sample_edges, upsample_factor, nu=nu, **kwargs).reshape(
            (-1, *np.ones(ndim - g.ndim, int), *g.shape)
        )
        for g in generators
    )


def sum_antiderivative(
    generators: Sequence[FunctionGenerator],
    n: int,
    n0: int = 0,
    sample_edges: bool = False,
    upsample_factor: int = 0,
    nu: int = 1,
    *,
    bias: float | NDArray | None = None,
    **kwargs,
) -> int:

    ref = max(generators, key=lambda g: g.ndim)
    ndim = ref.ndim

    if bias is not None:
        t = generators[0].ts(n, n0, sample_edges, upsample_factor).ravel()
        x = poly_antiderivative([bias], t, nu)

    return builtins.sum(
        (
            g.antiderivative(
                n, n0, sample_edges, upsample_factor, nu=nu, **kwargs
            ).reshape((-1, *np.ones(ndim - g.ndim, int), *g.shape))
            for g in generators
        ),
        start=0.0 if bias is None else x,
    )


setattr(sum, "derivative", sum_derivative)
setattr(sum, "antiderivative", sum_antiderivative)
