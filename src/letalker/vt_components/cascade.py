from itertools import chain as _chain
from typing import cast

import control as ct
import numpy as np
from scipy.linalg import block_diag


def block_antidiag(a1, a2):
    return np.fliplr(block_diag(np.fliplr(a1), np.fliplr(a2)))


def chain(*systems: *tuple[ct.LTI], **kwargs) -> ct.StateSpace:
    """cascade a chain of branch-less two-port systems

    Parameters
    ----------
    *systems
        N systems to be chained in the presented order. Every system
        must have the forward ports as the first input and output
        and the backward ports as the second input and output.

    Returns
    -------
        A chain of cascaded systems. F2 output and B2 input of the preceding
        system are connected to F1 input and B1 output of the next system.

        By default, the inputs of the overall system are F1 of the first system
        and B2 of the last system and outputs include F2 of the last system and
        B1 of the last system. If there is any auxiliary inputs or outputs, they
        are automatically appended to the overall input or output lists.

        These default inputs and outputs can be overridden by providing ``inplist``
        or ``outlist`` parameters of ``ct.interconnect`` function.

    To connect a system with more than 2 ports (e.g., LosslessJunction with more
    one inlet or outlet) use cascade() and specify the connecting inlet and outlet
    id.

    """

    # TODO: Add support for aux in/out when these classes are implemented

    if "inplist" not in kwargs:
        kwargs["inplist"] = [(0, "F1"), (len(systems) - 1, "B2")]
        if "inputs" not in kwargs:
            kwargs["inputs"] = ["F1", "B2"]
    if "outlist" not in kwargs:
        kwargs["outlist"] = [(len(systems) - 1, "F2"), (0, "B1")]
        if "outputs" not in kwargs:
            kwargs["outputs"] = ["F2", "B1"]

    return ct.interconnect(
        systems,
        connections=[
            *_chain(
                *(
                    [[(i + 1, "F1"), (i, "F2")], [(i, "B2"), (i + 1, "B1")]]
                    for i in range(len(systems) - 1)
                )
            )
        ],
        **kwargs,
    )


def cascade(
    sys1: ct.LTI, sys2: ct.LTI, sys1_outlet: int = 2, sys2_inlet: int = 1, **kwargs
) -> ct.StateSpace:
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

    f2 = f"F{sys1_outlet}"  # out
    b2 = f"B{sys1_outlet}"  # in
    f1 = f"F{sys2_inlet}"  # in
    b1 = f"B{sys2_inlet}"  # out

    if "inplist" not in kwargs:
        kwargs["inplist"] = [
            *((0, label) for label in sys1.input_labels if label != b2),
            *((1, label) for label in sys2.input_labels if label != f1),
        ]
        if "inputs" not in kwargs:
            kwargs["inputs"] = [label for (_, label) in kwargs["inplist"]]
    if "outlist" not in kwargs:
        kwargs["outlist"] = [
            *((1, label) for label in sys2.output_labels if label != b1),
            *((0, label) for label in sys1.output_labels if label != f2),
        ]
        if "outputs" not in kwargs:
            kwargs["outputs"] = [label for (_, label) in kwargs["outlist"]]

    return ct.interconnect(
        [sys1, sys2],
        connections=[[(1, f1), (0, f2)], [(0, b2), (1, b1)]],
        **kwargs,
    )


def combine(
    sys1: ct.LTI, sys2: ct.LTI, sys1_outlet: int = 2, sys2_inlet: int = 1, **kwargs
) -> ct.StateSpace:
    """combine two wave-reflection vocal tract subsystems by cascading them

    This function is similar to ``cascade`` but create a new state-space model

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

    assert sys1.dt is None or sys2.dt is None or np.isclose(sys1.dt, sys2.dt)

    f2 = f"F{sys1_outlet}"  # out
    b2 = f"B{sys1_outlet}"  # in
    f1 = f"F{sys2_inlet}"  # in
    b1 = f"B{sys2_inlet}"  # out

    if "inplist" not in kwargs:
        kwargs["inplist"] = [
            *((0, label) for label in sys1.input_labels if label != b2),
            *((1, label) for label in sys2.input_labels if label != f1),
        ]
        if "inputs" not in kwargs:
            kwargs["inputs"] = [label for (_, label) in kwargs["inplist"]]
    if "outlist" not in kwargs:
        kwargs["outlist"] = [
            *((1, label) for label in sys2.output_labels if label != b1),
            *((0, label) for label in sys1.output_labels if label != f2),
        ]
        if "outputs" not in kwargs:
            kwargs["outputs"] = [label for (_, label) in kwargs["outlist"]]

    ss1 = cast(ct.StateSpace, sys1.to_ss())
    ss2 = cast(ct.StateSpace, sys2.to_ss())

    def split(ss: ct.StateSpace, out_label: str, in_label: str) -> tuple:

        outp = ss.find_output(out_label)
        inp = ss.find_input(in_label)

        nst, nin = ss.B.shape
        nout = ss.noutputs

        Bt = np.empty((nst, nin - 1))
        Ct = np.empty((nout - 1, nst))
        d1t = np.empty((1, nin - 1))
        dt1 = np.empty((nout - 1, 1))
        Dt = np.empty((nout - 1, nin - 1))

        Bt[:, :inp], b1, Bt[:, inp:] = np.hsplit(ss.B, [inp, inp + 1])
        Ct[:outp, :], c1, Ct[outp:, :] = np.vsplit(ss.C, [outp, outp + 1])
        (
            (Dt[:outp, :inp], dt1[:outp, :], Dt[:outp, inp:]),
            (d1t[:, :inp], d11, d1t[:, inp:]),
            (Dt[outp:, :inp], dt1[outp:, :], Dt[outp:, inp:]),
        ) = (np.hsplit(dd, [inp, inp + 1]) for dd in np.vsplit(ss.D, [outp, outp + 1]))

        return b1, c1, d11, ss.A, Bt, Ct, d1t, dt1, Dt

    b1b, c1f, d1fb, A1, B1t, C1t, d1f, d1b, D1t = split(ss1, f2, b2)
    b2f, c2b, d2bf, A2, B2t, C2t, d2b, d2f, D2t = split(ss2, b1, f1)

    Bt = block_antidiag(b1b, b2f)
    Dt = block_diag(d2f, d1b)

    Q = np.array([[1, -d1fb[0, 0]], [-d2bf[0, 0], 1]])
    Cc = np.linalg.solve(Q, block_diag(c1f, c2b))
    Dc = np.linalg.solve(Q, block_diag(d1f, d2b))

    A = block_diag(A1, A2) + Bt @ Cc
    B = block_diag(B1t, B2t) + Bt @ Dc
    C = block_antidiag(C2t, C1t) + Dt @ Cc
    D = block_antidiag(D2t, D1t) + Dt @ Dc

    sys = cast(ct.StateSpace, ct.ss(A, B, C, D, dt=ss1.dt))

    # label the signals
    names1in = [name for i, name in enumerate(ss1.input_labels) if name != b2]
    names1out = [name for i, name in enumerate(ss1.output_labels) if name != f2]
    names2in = [name for i, name in enumerate(ss2.input_labels) if name != f1]
    names2out = [name for i, name in enumerate(ss2.output_labels) if name != b1]
    sys.update_names(inputs=[*names1in, *names2in], outputs=[*names2out, *names1out])

    return sys
