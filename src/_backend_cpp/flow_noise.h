#pragma once

#include <vector>

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h> // Required for nb::ndarray

#include "runner_base.h"

namespace nb = nanobind;

using namespace nb::literals;

struct NoFlowNoiseRunner : FlowNoiseRunnerBase
{
    double step(const unsigned int i, const double uin, const std::list<double> geom) override
    {
        return 0.0;
    }
};

struct LeTalkerAspirationNoiseRunner : FlowNoiseRunnerBase
{
    int n;
    const dbl_1darray nuL_inv;
    const dbl_1darray nf;
    const double re2b;

    unsigned int n_done;
    std::vector<double> re2;
    std::vector<double> ug_noise;

    LeTalkerAspirationNoiseRunner(
        const int nb_steps,
        const nb::object *s,
        const dbl_1darray nuL_inv_in,
        const dbl_1darray nf_in,
        const double RE2b_in) : n(nb_steps), nuL_inv(std::move(nuL_inv_in)),
                                nf(std::move(nf_in)), re2b(RE2b_in),
                                n_done(0), re2(nb_steps), ug_noise(nb_steps) {}

    double step(const unsigned int i, const double uin, const std::list<double> geom) override
    {
        unsigned offset;
        double nuL_inv_i, nf_i, RE2, u;

        nuL_inv_i = *DATA_PTR(nuL_inv, i);
        nf_i = *DATA_PTR(nf, i);

        RE2 = (uin * nuL_inv_i);
        RE2 *= RE2;
        u = (RE2 > re2b ? (RE2 - re2b) * nf_i : 0.0);

        if (i < n)
        {
            re2[i] = RE2;
            ug_noise[i] = u;
            n_done = i + 1;
        }
        return u;
    }

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_re2()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(re2.data(), {n_done});
    }

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_ug_noise()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(ug_noise.data(), {n_done});
    }
};

void bind_flow_noise(nb::module_ &m)
{
    nb::class_<NoFlowNoiseRunner, FlowNoiseRunnerBase>(m, "NoFlowNoiseRunner")
        .def(nb::init<>())
        .def("step", &NoFlowNoiseRunner::step, "i"_a, "u_in"_a, "geom"_a);

    nb::class_<LeTalkerAspirationNoiseRunner, FlowNoiseRunnerBase>(m, "LeTalkerAspirationNoiseRunner")
        .def(nb::init<const unsigned int, const nb::object *, const dbl_1darray, const dbl_1darray, const double>(),
             "nb_steps"_a, "sin"_a, "nuL_inv"_a, "nf"_a, "RE2b"_a)
        .def("step", &LeTalkerAspirationNoiseRunner::step, "i"_a, "u_in"_a, "geom"_a)
        .def_ro("n", &LeTalkerAspirationNoiseRunner::n_done)
        .def_ro("nuL_inv", &LeTalkerAspirationNoiseRunner::nuL_inv)
        .def_ro("nf", &LeTalkerAspirationNoiseRunner::nf)
        .def_ro("re2b", &LeTalkerAspirationNoiseRunner::re2b)
        .def_prop_ro("re2", &LeTalkerAspirationNoiseRunner::get_re2)
        .def_prop_ro("ug_noise", &LeTalkerAspirationNoiseRunner::get_ug_noise);
}
