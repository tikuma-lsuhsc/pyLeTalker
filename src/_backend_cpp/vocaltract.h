#pragma once

#include <vector>
#include <list>
#include <cmath>

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h> // Required for nb::ndarray

#include "runner_base.h"

namespace nb = nanobind;

using namespace nb::literals;

struct WaveReflectionVocalTractFastRunner : RunnerBase
{

    const dbl_2darray m_alph_even;
    const dbl_2darray m_alph_odd;
    const dbl_2darray m_r_even;
    const dbl_2darray m_r_odd;

    WaveReflectionVocalTractFastRunner(const unsigned nb_steps,
                                       dbl_1darray s_in,
                                       const dbl_2darray alph_odd,
                                       const dbl_2darray alph_even,
                                       const dbl_2darray r_odd,
                                       const dbl_2darray r_even) : m_alph_even(std::move(alph_even)),
                                                                  m_alph_odd(std::move(alph_odd)),
                                                                  m_r_even(std::move(r_even)),
                                                                  m_r_odd(std::move(r_odd)),
                                                                  n_even(r_even.shape(1)),
                                                                  n_odd(r_odd.shape(1)),
                                                                  s(std::move(s_in)),
                                                                  s_even(s.data(), n_even),
                                                                  s_odd(s.data() + n_even, n_even),
                                                                  f_odd(n_odd), b_even(n_odd), 
                                                                  Psi_even(n_even), Psi_odd(n_odd)
    {
    }

    pressure_pair step(const unsigned int i, const double f1, const double bK) override
    {
        // get the current vocal tract attenuation and reflection coefficients
        Eigen::Map<Eigen::ArrayXd> alph_even(DATA_PTR(m_alph_even, i), n_odd);
        Eigen::Map<Eigen::ArrayXd> alph_odd(DATA_PTR(m_alph_odd, i), n_odd);
        Eigen::Map<Eigen::ArrayXd> r_even(DATA_PTR(m_r_even, i), n_even);
        Eigen::Map<Eigen::ArrayXd> r_odd(DATA_PTR(m_r_odd, i), n_odd);

        // ---even junctions [(F2,B3),(F4,B5),...,(FK-2,BK-1)]->[(F3,B2),(F5,B4),...]--- */
        Psi_even = (s_even - s_odd) * r_even;
        f_odd(0) = f1;
        f_odd(Eigen::seqN(1, n_even)) = s_even + Psi_even;
        b_even(Eigen::seqN(0, n_even)) = s_odd + Psi_even;
        b_even(n_even) = bK;

        // propagating over a tube segment
        f_odd *= alph_odd;
        b_even *= alph_even;

        // ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */
        Psi_odd = (f_odd - b_even) * r_odd; //[F3, F5, ...] - [ B2, B4, ... ]
        f_odd += Psi_odd;
        b_even += Psi_odd;

        // propagating over the other tube segment --- attenuate pressure to "pre-"propagate signal to the output of the section
        f_odd *= alph_even;
        b_even *= alph_odd;

        s_even = f_odd(Eigen::seqN(0, n_even));
        s_odd = b_even(Eigen::seqN(1, n_even));

        return pressure_pair(f_odd(n_even), b_even(0));
    }

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_sout()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(s.data(), {2 * n_even});
    }

    private:

    //array lengths
    unsigned n_even; // each state
    unsigned n_odd; // n_even + 1

    // state variables
    dbl_1darray s;
    Eigen::Map<Eigen::ArrayXd> s_even;
    Eigen::Map<Eigen::ArrayXd> s_odd;

    // temp variables
    Eigen::ArrayXd f_odd;
    Eigen::ArrayXd b_even;
    Eigen::ArrayXd Psi_even;
    Eigen::ArrayXd Psi_odd;

};

void bind_vocaltract(nb::module_ &m)
{
    nb::class_<WaveReflectionVocalTractFastRunner, RunnerBase>(m, "WaveReflectionVocalTractFastRunner")
        .def(nb::init<const unsigned, const dbl_1darray, const dbl_2darray,
                      const dbl_2darray, const dbl_2darray, const dbl_2darray>(),
             "nb_steps"_a, "sin"_a, "alph_even"_a, "alph_odd"_a, "r_even"_a, "r_odd"_a)
        .def("step", &WaveReflectionVocalTractFastRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_prop_ro("sout", &WaveReflectionVocalTractFastRunner::get_sout, nb::rv_policy::reference_internal);
}
