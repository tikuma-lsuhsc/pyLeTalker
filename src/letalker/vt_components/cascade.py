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


def _cascade(sys1: ct.LTI, sys2: ct.LTI) -> ct.StateSpace:
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

    A1 = ss1.A
    B1 = ss1.B[:, :nin1]
    C1 = ss1.C[:nout1, :]
    D1 = ss1.D[:nout1, :nin1]

    A2 = ss2.A
    B2 = ss2.B[:, :nin2]
    C2 = ss2.C[:nout2, :]
    D2 = ss2.D[:nout2, :nin2]

    gamma = 1 - D1[0, 1] * D2[1, 0]

    U = np.array([[1, 0], [0, 0]])
    L = np.array([[0, 0], [0, 1]])
    I = np.eye(2)

    Q1 = I - D1 @ L @ D2 @ U
    Q2 = I - D2 @ U @ D1 @ L

    P1 = D1 @ (L @ D2 @ L + U)
    P2 = D2 @ (U @ D1 @ U + L)

    LQ1 = L @ np.linalg.inv(Q1)
    LQ2 = L / gamma  # L @ np.linalg.inv(Q2)
    UQ1 = U / gamma  # U @ np.linalg.inv(Q1)
    UQ2 = U @ np.linalg.inv(Q2)

    # I = np.eye(4)
    # np.linalg.block_diag(D1@L,D2@U)

    C = np.block([(LQ1 + UQ2 @ D2 @ U) @ C1, (UQ2 + LQ1 @ D1 @ L) @ C2])
    D = UQ2 @ P2 + LQ1 @ P1

    A = np.block(
        [
            [A1 + B1 @ LQ2 @ D2 @ U @ C1, B1 @ LQ2 @ C2],
            [B2 @ UQ1 @ C1, A2 + B2 @ UQ1 @ D1 @ L @ C2],
        ]
    )
    B = np.block([[B1 @ (LQ2 @ P2 + U)], [B2 @ (UQ1 @ P1 + L)]])

    # b12 = B1[:, 1:2]
    # b21 = B2[:, 0:1]
    # c11 = C1[0:1, :]
    # c22 = C2[1:2, :]
    # (d111, d112), (d121, d122) = D1
    # (d211, d212), (d221, d222) = D2

    # A = np.zeros((nst, nst))
    # A[:nst1, :nst1] = A1 + (b12 * d221 / gamma) @ c11
    # A[:nst1, nst1:] = (b12 / gamma) @ c22
    # A[nst1:, :nst1] = (b21 / gamma) @ c11
    # A[nst1:, nst1:] = A2 + (b21 * d112) @ c22

    # B = np.zeros((nst, nin))
    # B[:nst1, :nout] = B1 @ np.array([[1, 0], [d111 * d221 / gamma, d222 / gamma]])
    # B[nst1:, :nout] = B2 @ np.array([[d111 / gamma, d112 * d222 / gamma], [0, 1]])
    # if naux1:
    #     B[:nst1, nout:-naux2] = ss1.B[:, nout:]
    # if naux2:
    #     B[nst1:, -naux2:] = ss2.B[:, nout:]

    # K1 = np.array([[d211 / gamma, 0], [d122 * d221 / gamma, 1]])
    # K2 = np.array([[1, d112 * d211 / gamma], [0, d122 / gamma]])
    # C = np.zeros((nout, nst))
    # C[:, :nst1] = K1 @ C1
    # C[:, nst1:] = K2 @ C2

    # D = np.zeros((nout, nin))
    # D[0, :nout] = [d111 * d211 / gamma, d112 * d211 * d222 / gamma + d212]
    # D[1, :nout] = [d111 * d122 * d221 / gamma + d121, d122 * d222 / gamma]

    # if naux1:
    #     D[0, nout:-naux2] = K1 @ ss1.D[:, nout:]
    # if naux2:
    #     D[1, -naux2:] = K2 @ ss2.D[:, nout:]

    return ct.ss(A, B, C, D, dt=ss1.dt)
