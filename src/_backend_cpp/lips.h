#pragma once

#include <vector>

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h> // Required for nb::ndarray

#include "runner_base.h"

namespace nb = nanobind;

using namespace nb::literals;

struct LeTalkerLipsRunner : RunnerBase
{

    LeTalkerLipsRunner(const unsigned int nb_steps,
                       Eigen::Vector3d s_in,
                       const dbl_ndarray<nb::shape<-1, 2, 3>> A_in,
                       const dbl_ndarray<nb::shape<-1, 2>> b_in,
                       const dbl_1darray c_in) : n(nb_steps), s(std::move(s_in)),
                                                 A(std::move(A_in)), b(std::move(b_in)),
                                                 c(std::move(c_in)), n_done(0),
                                                 pout(std::vector<double>(nb_steps)),
                                                 uout(std::vector<double>(nb_steps)) {}

    pressure_pair step(const unsigned int i, const double fin, const double bin) override
    {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Ai(DATA_PTR(A, i));
        Eigen::Map<Eigen::Vector2d> bi(DATA_PTR(b, i));
        double ci = *DATA_PTR(c, i);

        s(Eigen::seq(0, 1)) = Ai * s + bi * fin;
        s(2) = fin;

        if (i < n)
        {
            pout[i] = s(0);
            uout[i] = ci * (fin - s(1));
        }

        n_done = i + 1;

        return pressure_pair(0.0, s(1));
    }

    unsigned int n;
    const dbl_3darray A;
    const dbl_2darray b;
    const dbl_1darray c;

    unsigned int n_done;
    Eigen::Vector3d s;
    std::vector<double> pout;
    std::vector<double> uout;

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_pout()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(pout.data(), {n_done});
    }

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_uout()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(uout.data(), {n_done});
    }
};

void bind_lips(nb::module_ &m)
{
    nb::class_<LeTalkerLipsRunner, RunnerBase>(m, "LeTalkerLipsRunner")
        .def(nb::init<const unsigned int, Eigen::Vector3d, const dbl_ndarray<nb::shape<-1, 2, 3>>,
                      const dbl_ndarray<nb::shape<-1, 2>>, const dbl_1darray>(),
             "nb_steps"_a, "sin"_a, "A"_a, "b"_a, "c"_a)
        .def("step", &LeTalkerLipsRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("n", &LeTalkerLipsRunner::n_done, nb::rv_policy::reference_internal)
        .def_ro("sout", &LeTalkerLipsRunner::s, nb::rv_policy::reference_internal)
        .def_prop_ro("pout", &LeTalkerLipsRunner::get_pout, nb::rv_policy::reference_internal)
        .def_prop_ro("uout", &LeTalkerLipsRunner::get_uout, nb::rv_policy::reference_internal);
}
