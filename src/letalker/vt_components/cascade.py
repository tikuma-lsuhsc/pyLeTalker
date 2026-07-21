from itertools import chain as _chain

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
