#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/trampoline.h>

namespace nb = nanobind;

using namespace nb::literals;

typedef std::pair<double, double> pressure_pair;

struct RunnerBase
{
    virtual ~RunnerBase() = default;
    virtual pressure_pair step(const int i, const double fin, const double bin) = 0;
};

struct PyRunnerBase : RunnerBase
{
    NB_TRAMPOLINE(RunnerBase, 1);

    virtual pressure_pair step(const int i, const double fin, const double bin) override
    {
        NB_OVERRIDE_PURE(step, i, fin, bin);
    }
};

struct LeTalkerLungsRunner : RunnerBase
{
    LeTalkerLungsRunner(const int nb_steps,
                        const nb::object *s,
                        const nb::ndarray<double, nb::ndim<1>, nb::c_contig> p_lungs) : n(nb_steps),
                                                                                        plung(std::move(p_lungs)) {}

    virtual pressure_pair step(const int i, const double fin, const double bin) override
    {
        int offset = i < plung.shape(0) ? i : plung.shape(0) - 1;
        double lung_pressure = plung.data()[offset];
        return pressure_pair(.9 * lung_pressure - 0.8 * bin, 0.0);
    }

    const int n;
    const nb::ndarray<double, nb::ndim<1>, nb::c_contig> plung;
};

struct OpenLungsRunner : RunnerBase
{
    OpenLungsRunner(const int nb_steps,
                    const nb::object *s,
                    const nb::ndarray<double, nb::ndim<1>, nb::c_contig> p_lungs) : n(nb_steps),
                                                                                    plung(std::move(p_lungs)) {}

    pressure_pair step(const int i, const double fin, const double bin) override
    {
        int offset = i < plung.shape(0) ? i : plung.shape(0) - 1;
        double lung_pressure = plung.data()[offset * plung.stride(0)];
        return pressure_pair(lung_pressure, 0.0);
    }

    const int n;
    const nb::ndarray<double, nb::ndim<1>, nb::c_contig> plung;
};

struct NullLungsRunner : RunnerBase
{
    NullLungsRunner(const int nb_steps,
                    const nb::object *s) : n(nb_steps) {}
    pressure_pair step(const int i, const double fin, const double bin) override
    {
        return pressure_pair(-0.8 * bin, 0.0);
    }

    const int n;
};

void sim_loop(int nb_steps, RunnerBase &lungs, RunnerBase &trachea, RunnerBase &vf, RunnerBase &vt, RunnerBase &lips)
{
    double flung, blung, fsg, bsg, feplx, beplx, flip, blip;

    flung = blung = fsg = bsg = feplx = beplx = flip = blip = 0.0;

    for (int i = 0; i < nb_steps; ++i)
    {
        // Compute current pressure outputs from VOCAL FOLD
        pressure_pair out = vf.step(i, fsg, beplx);
        feplx = std::get<0>(out);
        bsg = std::get<1>(out);

        // Compute the next states of flip & beplx
        out = vt.step(i, feplx, blip);
        flip = std::get<0>(out);
        beplx = std::get<1>(out);

        out = lips.step(i, flip, 0.0);
        blip = std::get<1>(out);

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
        ``Lips``: ``fout`` is ignored by the simulator)"
)
        .def(nb::init<>());
    nb::class_<LeTalkerLungsRunner, RunnerBase>(m, "LeTalkerLungsRunner")
        .def(nb::init<const int, const nb::object *, const nb::ndarray<double, nb::ndim<1>, nb::c_contig>>(), "nb_steps"_a, "s_in"_a, "p_lungs"_a)
        .def("step", &LeTalkerLungsRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &LeTalkerLungsRunner::n)
        .def_ro("plung", &LeTalkerLungsRunner::plung);
    nb::class_<OpenLungsRunner, RunnerBase>(m, "OpenLungsRunner")
        .def(nb::init<const int, const nb::object *, const nb::ndarray<double, nb::ndim<1>, nb::c_contig>>(), "nb_steps"_a, "s_in"_a, "p_lungs"_a)
        .def("step", &OpenLungsRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &OpenLungsRunner::n)
        .def_ro("plung", &OpenLungsRunner::plung);
    nb::class_<NullLungsRunner, RunnerBase>(m, "NullLungsRunner")
        .def(nb::init<const int, const nb::object *>(), "nb_steps"_a, "s_in"_a)
        .def("step", &NullLungsRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &NullLungsRunner::n);
    m.def("sim_loop", &sim_loop, nb::call_guard<nb::gil_scoped_release>());
}
