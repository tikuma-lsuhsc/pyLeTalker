#pragma once

#include <vector>
#include <list>

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h> // Required for nb::ndarray

#include "runner_base.h"

namespace nb = nanobind;

using namespace nb::literals;

struct VocalFoldsUgRunner : RunnerBase
{

    FlowNoiseRunnerBase &anoise;

    unsigned n;
    dbl_1darray v_ug;
    dbl_1darray v_rhoca_sg;
    dbl_1darray v_rhoca_eplx;

    unsigned n_done;
    std::vector<double> v_psg;
    std::vector<double> v_peplx;

    VocalFoldsUgRunner(const unsigned int nb_steps,
                       const nb::object s_in,
                       FlowNoiseRunnerBase &anoise_in,
                       const dbl_1darray ug,
                       const dbl_1darray rhoca_sg,
                       const dbl_1darray rhoca_eplx) : n(nb_steps), anoise(anoise_in),
                                                       v_ug(std::move(ug)),
                                                       v_rhoca_sg(std::move(rhoca_sg)),
                                                       v_rhoca_eplx(std::move(rhoca_eplx)),
                                                       n_done(0),
                                                       v_psg(std::vector<double>(nb_steps)),
                                                       v_peplx(std::vector<double>(nb_steps)) {}

    pressure_pair step(const unsigned int i, const double fsg, const double beplx) override
    {
        double ug, rhoca_eplx, rhoca_sg, feplx, bsg;

        rhoca_eplx = *DATA_PTR(v_rhoca_eplx, i);
        rhoca_sg = *DATA_PTR(v_rhoca_sg, i);
        ug = *DATA_PTR(v_ug, i) + anoise.step(i, ug, std::list<double>());

        feplx = beplx + ug * rhoca_eplx;
        bsg = fsg - ug * rhoca_sg;

        if (i < n)
        {
            v_psg[i] = fsg + bsg;
            v_peplx[i] = feplx + beplx;
        }
        n_done = i + 1;

        return pressure_pair(feplx, bsg);
    }

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_psg()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(v_psg.data(), {n_done});
    }

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_peplx()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(v_peplx.data(), {n_done});
    }
};

void bind_vocalfolds(nb::module_ &m)
{
    nb::class_<VocalFoldsUgRunner, RunnerBase>(m, "VocalFoldsUgRunner")
        .def(nb::init<const unsigned, const nb::object, FlowNoiseRunnerBase &,
                      const dbl_1darray, const dbl_1darray, const dbl_1darray>(),
             "nb_steps"_a, "sin"_a, "anoise"_a, "ug"_a, "rhoca_sg"_a, "rhoca_eplx"_a)
        .def("step", &VocalFoldsUgRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &VocalFoldsUgRunner::n_done, nb::rv_policy::reference_internal)
        .def_ro("ug", &VocalFoldsUgRunner::v_ug, nb::rv_policy::reference_internal)
        .def_prop_ro("psg", &VocalFoldsUgRunner::get_psg, nb::rv_policy::reference_internal)
        .def_prop_ro("peplx", &VocalFoldsUgRunner::get_peplx, nb::rv_policy::reference_internal);
}
