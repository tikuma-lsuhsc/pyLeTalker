#pragma once

#include <nanobind/nanobind.h>

#include "runner_base.h"

namespace nb = nanobind;

using namespace nb::literals;

struct LeTalkerLungsRunner : RunnerBase
{
    LeTalkerLungsRunner(const unsigned int nb_steps,
                        const nb::object *s,
                        const dbl_1darray p_lungs) : n(nb_steps),
                                                     plung(std::move(p_lungs)) {}

    virtual pressure_pair step(const unsigned int i, const double fin, const double bin) override
    {
        int offset = i < plung.shape(0) ? i : plung.shape(0) - 1;
        double lung_pressure = plung.data()[offset];
        return pressure_pair(.9 * lung_pressure - 0.8 * bin, 0.0);
    }

    const unsigned int n;
    const dbl_1darray plung;
};

struct OpenLungsRunner : RunnerBase
{
    OpenLungsRunner(const unsigned int nb_steps,
                    const nb::object *s,
                    const dbl_1darray p_lungs) : n(nb_steps),
                                                 plung(std::move(p_lungs)) {}

    pressure_pair step(const unsigned int i, const double fin, const double bin) override
    {
        int offset = i < plung.shape(0) ? i : plung.shape(0) - 1;
        double lung_pressure = plung.data()[offset * plung.stride(0)];
        return pressure_pair(lung_pressure, 0.0);
    }

    const unsigned int n;
    const dbl_1darray plung;
};

struct NullLungsRunner : RunnerBase
{
    NullLungsRunner(const unsigned int nb_steps,
                    const nb::object *s) : n(nb_steps) {}
    pressure_pair step(const unsigned int i, const double fin, const double bin) override
    {
        return pressure_pair(-0.8 * bin, 0.0);
    }

    const unsigned int n;
};

void bind_lungs(nb::module_ &m)
{
    nb::class_<LeTalkerLungsRunner, RunnerBase>(m, "LeTalkerLungsRunner")
        .def(nb::init<const unsigned int, const nb::object *, const dbl_1darray>(), "nb_steps"_a, "s_in"_a, "p_lungs"_a)
        .def("step", &LeTalkerLungsRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &LeTalkerLungsRunner::n)
        .def_ro("plung", &LeTalkerLungsRunner::plung)
        .def_prop_ro("sout", [](){ return nb::none(); });
    nb::class_<OpenLungsRunner, RunnerBase>(m, "OpenLungsRunner")
        .def(nb::init<const unsigned int, const nb::object *, const dbl_1darray>(), "nb_steps"_a, "s_in"_a, "p_lungs"_a)
        .def("step", &OpenLungsRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &OpenLungsRunner::n)
        .def_ro("plung", &OpenLungsRunner::plung)
        .def_prop_ro("sout", [](){ return nb::none(); });
    nb::class_<NullLungsRunner, RunnerBase>(m, "NullLungsRunner")
        .def(nb::init<const unsigned int, const nb::object *>(), "nb_steps"_a, "s_in"_a)
        .def("step", &NullLungsRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &NullLungsRunner::n)
        .def_prop_ro("sout", [](){ return nb::none(); });
}
