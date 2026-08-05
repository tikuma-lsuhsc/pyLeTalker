from typing import Literal

import control as ct
import numpy as np
from scipy.signal import cont2discrete


def sample(
    sys: ct.LTI,
    Ts: float,
    method: Literal[
        "gbt", "bilinear", "euler", "backward_diff", "zoh", "foh", "impulse"
    ] = "bilinear",
    alpha: float | None = None,
    prewarp_frequency: float | None = None,
    name: str | None = None,
    copy_names: bool = True,
    **kwargs,
) -> ct.LTI:

    if not sys.isctime():
        raise ValueError("System must be continuous-time system")

    if prewarp_frequency is not None:
        if method not in ("bilinear", "tustin") or (method == "gbt" and alpha == 0.5):
            raise ValueError(
                "prewarp_frequency cannot be used with %s: incompatible conversion"
            )

        Twarp = 2 * np.tan(prewarp_frequency * Ts / 2) / prewarp_frequency

    is_tf = isinstance(sys, ct.TransferFunction)

    if is_tf:
        sys = sys.to_ss()

    Twarp = (
        Ts
        if prewarp_frequency is None
        else 2 * np.tan(prewarp_frequency * Ts / 2) / prewarp_frequency
    )

    Ad, Bd, C, D, _ = cont2discrete((sys.A, sys.B, sys.C, sys.D), Twarp, method, alpha)

    # pass desired signal names if names were provided
    sysd = ct.StateSpace(Ad, Bd, C, D, Ts, **kwargs)

    if is_tf:
        sysd = sysd.to_tf(**kwargs)

    # copy over the system name, inputs, outputs, and states
    if copy_names:
        sysd._copy_names(sys, prefix_suffix_name="sampled")
    if name is not None:
        sysd.name = name

    return sysd
