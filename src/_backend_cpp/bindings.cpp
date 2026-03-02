#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include "runner_base.h"
#include "lungs.h"
#include "lips.h"

namespace nb = nanobind;

using namespace nb::literals;

struct PyRunnerBase : RunnerBase
{
    NB_TRAMPOLINE(RunnerBase, 1);

    virtual pressure_pair step(const unsigned int i, const double fin, const double bin) override
    {
        NB_OVERRIDE_PURE(step, i, fin, bin);
    }
};


struct PyFlowNoiseRunnerBase : FlowNoiseRunnerBase
{
    NB_TRAMPOLINE(FlowNoiseRunnerBase, 1);

    virtual double step(const unsigned int i, const double uin, const std::list<double> geom) override
    {
        NB_OVERRIDE_PURE(step, i, uin, geom);
    }
};

void sim_loop(int nb_steps,
              RunnerBase &lungs,
              RunnerBase &trachea,
              RunnerBase &vf,
              RunnerBase &vt,
              RunnerBase &lips)
{
    double flung, blung, fsg, bsg, feplx, beplx, fin, bin;

    flung = blung = fsg = bsg = feplx = beplx = fin = bin = 0.0;

    for (int i = 0; i < nb_steps; ++i)
    {
        // Compute current pressure outputs from VOCAL FOLD
        pressure_pair out = vf.step(i, fsg, beplx);
        feplx = std::get<0>(out);
        bsg = std::get<1>(out);

        // Compute the next states of fin & beplx
        out = vt.step(i, feplx, bin);
        fin = std::get<0>(out);
        beplx = std::get<1>(out);

        out = lips.step(i, fin, 0.0);
        bin = std::get<1>(out);

        // Compute the next states of fsg & blung
        out = lungs.step(i, 0.0, blung);
        flung = std::get<0>(out);

        out = trachea.step(i, flung, bsg);
        fsg = std::get<0>(out);
        blung = std::get<1>(out);
    }
}

NB_MODULE(_backend, m)
{
    nb::class_<RunnerBase>(m, "RunnerBase");
    nb::class_<PyRunnerBase, RunnerBase>(m, "PyRunnerBase", R"(
Base class for the ``Runner`` classes of every ``Element`` class

To run the synthesis simulation, every ``Element`` object (of type 
``ElementClass``) will create a runner object of type ``ElementClass.Runner``.
A ``Runner`` class may be written in C++ or in Python. ``PyRunnerBase`` is the 
required superclass of a Python runner class. 

To subclass ``PyRunnerBase``, there are two requirements:

1. ``super().__init__()`` must be called during subclass initialization
2. ``step()`` method must be implemented with the following signature:

     ``def step(self, i: int, fin: float, bin: float) -> tuple[float, float]``

    This method performs the element's operation during one simulation step at 
    time ``i``. The following defines its arguments and outputs:

      Args:
        i: current simulation step
        fin: input forward traveling pressure
        bin: input backward traveling pressure

      Returns:
        fout: output forward traveling pressure
        bout: output backward traveling pressure

      Note:
        ``Lungs``: ``fin`` is always ``0.0``
        ``Lips``: ``fout`` is ignored by the simulator)")
        .def(nb::init<>());

    nb::class_<FlowNoiseRunnerBase>(m, "FlowNoiseRunnerBase");
    nb::class_<PyFlowNoiseRunnerBase, FlowNoiseRunnerBase>(m, "PyFlowNoiseRunnerBase", R"(
Base class for the ``Runner`` classes of every ``FlowNoise`` class

To run the synthesis simulation, every ``FlowNoise`` object (of type 
``FlowNoiseClass``) will create a runner object of type ``FlowNoiseClass.Runner``.
A ``Runner`` class may be written in C++ or in Python. ``PyFlowNoiseRunnerBase`` 
is the required superclass of a Python runner class. 

To subclass ``PyFlowNoiseRunnerBase``, there are two requirements:

1. ``super().__init__()`` must be called during subclass initialization
2. ``step()`` method must be implemented with the following signature:

     ``def step(self, i: int, uin: float, geom: list[float]) -> float``

    This method performs the element's operation during one simulation step at 
    time ``i``. The following defines its arguments and outputs:

      Args:
        i: current simulation step
        uin: input flow
        geom: runtime dynamic geometry parameters in a prescribed order 
            (empty list if none needed to be provided during simulation)

      Returns:
        flow noise to be injected)")
        .def(nb::init<>());

    m.def("sim_loop", &sim_loop, nb::call_guard<nb::gil_scoped_release>());

    bind_lungs(m);
    bind_lips(m);
}
