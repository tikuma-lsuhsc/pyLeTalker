import control as ct
import numpy as np


def cascade(sys1: ct.LTI, sys2: ct.LTI) -> ct.StateSpace:
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
    ss1 = ct.ss(sys1)
    ss2 = ct.ss(sys2)

    nst1, nst2 = ss1.nstates, ss2.nstates
    nin1, nin2 = ss1.ninputs, ss2.ninputs
    nout1, nout2 = ss1.noutputs, ss2.noutputs

    assert (
        nin1 is not None
        and nin2 is not None
        and nout1 == 2
        and nout2 == 2
        and nin1 >= 2
        and nin2 >= 2
    )

    naux1 = nin1 - nout1
    naux2 = nin2 - nout2
    nout = 2

    naux = naux1 + naux2
    nin = nout + naux
    nst = nst1 + nst2

    b12 = ss1.B[:, 1:2]
    b21 = ss2.B[:, 0:1]
    c11 = ss1.C[0:1, :]
    c22 = ss2.C[1:2, :]
    (d111, d112), (d121, d122) = ss1.D[:, :nout]
    (d211, d212), (d221, d222) = ss2.D[:, :nout]

    gamma = 1 - d112 * d221

    A = np.zeros((nst, nst))
    A[:nst1, :nst1] = ss1.A + (b12 * d221 / gamma) @ c11
    A[:nst1, nst1:] = (b12 / gamma) @ c22
    A[nst1:, :nst1] = (b21 / gamma) @ c11
    A[nst1:, nst1:] = ss2.A + (b21 * d112) @ c22

    B = np.zeros((nst, nin))
    B[:nst1, :nout] = ss1.B[:, :nout] @ np.array(
        [[1, 0], [d111 * d221 / gamma, d222 / gamma]]
    )
    B[nst1:, :nout] = ss2.B[:, :nout] @ np.array(
        [[d111 / gamma, d112 * d222 / gamma], [0, 1]]
    )
    if naux1:
        B[:nst1, nout:-naux2] = ss1.B[:, nout:]
    if naux2:
        B[nst1:, -naux2:] = ss2.B[:, nout:]

    K1 = np.array([[d211 / gamma, 0], [d122 * d221 / gamma, 1]])
    K2 = np.array([[1, d112 * d211 / gamma], [0, d122 / gamma]])
    C = np.zeros((nout, nst))
    C[:, :nst1] = K1 @ ss1.C
    C[:, :nst2] = K2 @ ss2.C

    D = np.zeros((nout, nin))
    D[0, :nout] = [d111 * d211 / gamma, d112 * d211 * d222 / gamma + d212]
    D[1, :nout] = [d111 * d122 * d221 / gamma + d121, d122 * d222 / gamma]

    if naux1:
        D[0, nout:-naux2] = K1 @ ss1.D[:, nout:]
    if naux2:
        D[1, -naux2:] = K2 @ ss2.D[:, nout:]

    return ct.ss(A, B, C, D)
