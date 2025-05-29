from __future__ import annotations

from numpy.typing import NDArray

import numpy as np


def align_signals(*signals: NDArray, ndim: int | None = None) -> tuple[NDArray]:
    """match signal dimensions for broadcasting

    Parameters
    ----------
    signals
        numpy arrays of generated signals. Their first axis is the time axis.
    ndim, optional
        explicit target array dimension

    Returns
    -------
        arrays with a fewer dimensions gets extra dimensions inserted before
        the time axis to match the highest dimensions among the signals.

    """

    if len(signals) <= 1:
        return signals

    if ndim is None:
        ndim = max(signals, key=lambda x: x.ndim).ndim

    return tuple(
        (
            x.reshape(x.shape[0], *np.ones(ndim - x.ndim, int), *x.shape[1:])
            if ndim > x.ndim
            else x
        )
        for x in signals
    )
