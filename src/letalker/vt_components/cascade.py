import control as ct
import numpy as np


def cascade(*systems: tuple[ct.LTI]) -> ct.StateSpace:
    """cascade two wave-reflection vocal tract subsystems

    Parameters
    ----------
    sys1
        leading subsystem, its first 2 outputs connects to sys2 inputs
    sys2
        following subsystem, its first 2 inputs connects to sys1 outputs

    Returns
    -------
        cascaded system
    """

    sys = systems[0]
    for s in systems[1:]:
        sys = _cascade(sys, s)
    return sys


def _cascade(
    sys1: ct.LTI,
    sys2: ct.LTI,
    fwd_out: int = 0,
    bwd_in: int = 1,
    bwd_out: int = 1,
    fwd_in: int = 0,
) -> ct.StateSpace:
    """cascade two wave-reflection vocal tract subsystems

    Parameters
    ----------
    sys1
        leading subsystem, its first 2 outputs connects to sys2 inputs
    sys2
        following subsystem, its first 2 inputs connects to sys1 outputs
    fwd_out, optional
        sys1 forward output port index, by default 0
    bwd_in, optional
        sys1 backward input port index, by default 1
    bwd_out, optional
        sys2 backward output port index, by default 1
    fwd_in, optional
        sys2 forward input port index, by default 0

    Returns
    -------
        cascaded system
    """
    ss1 = ct.ss(sys1)
    ss2 = ct.ss(sys2)

    nst1, nst2 = ss1.nstates, ss2.nstates
    nin1, nin2 = ss1.ninputs, ss2.ninputs
    nout1, nout2 = ss1.noutputs, ss2.noutputs

    nin1t, nin2t = nin1 - 1, nin2 - 1
    nout1t, nout2t = nout1 - 1, nout2 - 1

    assert (
        nin1 is not None
        and nin2 is not None
        and nout1 >= 2
        and nout2 >= 2
        and nin1 >= 2
        and nin2 >= 2
        and ss1.dt == ss2.dt
    )

    b1b = ss1.B[:, bwd_in : bwd_in + 1]
    b2f = ss2.B[:, fwd_in : fwd_in + 1]
    c1f = ss1.C[fwd_out : fwd_out + 1, :]
    c2b = ss2.C[bwd_out : bwd_out + 1, :]

    in1 = np.ones(nin1, bool)
    in1[bwd_in] = False
    in2 = np.ones(nin2, bool)
    in2[fwd_in] = False

    out1 = np.ones(nout1, bool)
    out1[fwd_out] = False
    out2 = np.ones(nout2, bool)
    out2[bwd_out] = False

    B1t = ss1.B[:, in1]
    B2t = ss2.B[:, in2]
    C1t = ss1.C[out1, :]
    C2t = ss2.C[out2, :]

    d1fb = ss1.D[fwd_out, bwd_in]
    d1f = ss1.D[fwd_out, in1]
    d1b = ss1.D[out1, bwd_in]
    D1t = ss1.D[out1, in1]
    d2bf = ss2.D[bwd_out, fwd_in]
    d2b = ss2.D[bwd_out, in2]
    d2f = ss2.D[out2, fwd_in]
    D2t = ss2.D[out2, in2]

    Qc = np.eye(2) - np.array([[0, d1fb], [d2bf, 0]])
    Cc = np.linalg.lstsq(
        Qc, np.block([[c1f, np.zeros((1, nst2))], [np.zeros((1, nst1)), c2b]])
    )[0]
    Dc = np.linalg.lstsq(
        Qc, np.block([[d1f, np.zeros((1, nin2t))], [np.zeros((1, nin1t)), d2b]])
    )[0]

    Bcc = np.block([[np.zeros((nst1, 1)), b1b], [b2f, np.zeros((nst2, 1))]])
    A = (
        np.block([[ss1.A, np.zeros((nst1, nst2))], [np.zeros((nst2, nst1)), ss2.A]])
        + Bcc @ Cc
    )
    B = (
        np.block([[B1t, np.zeros((nst1, nin2t))], [np.zeros((nst2, nin1t)), B2t]])
        + Bcc @ Dc
    )
    Dcc = np.block([[d2f, np.zeros((nout2t, 1))], [np.zeros((nout1t, 1)), d1b]])
    C = (
        np.block([[np.zeros((nout2t, nst1)), C2t], [C1t, np.zeros((nout1t, nst2))]])
        + Dcc @ Cc
    )
    D = (
        np.block([[np.zeros((nout2t, nin1t)), D2t], [D1t, np.zeros((nout1t, nin2t))]])
        + Dcc @ Dc
    )

    return ct.ss(A, B, C, D, dt=ss1.dt)
