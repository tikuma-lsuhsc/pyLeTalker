#pragma once

#include <vector>
#include <list>
#include <cmath>

#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h> // Required for nb::ndarray

#include "runner_base.h"

namespace nb = nanobind;

using namespace nb::literals;

struct WaveReflectionVocalTractRunner : RunnerBase
{

    const dbl_2darray m_alph_even;
    const dbl_2darray m_alph_odd;
    const dbl_2darray m_r_even;
    const dbl_2darray m_r_odd;

    unsigned n;
    unsigned int n_done;
    const bool b_log;

private:
    // array lengths
    unsigned n_even; // each state
    unsigned n_odd;  // n_even + 1

    // state variables
    dbl_1darray s;
    Eigen::Map<Eigen::ArrayXd> s_even;
    Eigen::Map<Eigen::ArrayXd> s_odd;

    // temp variables
    Eigen::ArrayXd f_odd;
    Eigen::ArrayXd b_even;
    Eigen::ArrayXd Psi_even;
    Eigen::ArrayXd Psi_odd;

    // log varaiable
    std::vector<double> p_sections;

    typedef Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>> SectionBlock;

public:
    WaveReflectionVocalTractRunner(const unsigned nb_steps,
                                   dbl_1darray s_in,
                                   const dbl_2darray alph_odd,
                                   const dbl_2darray alph_even,
                                   const dbl_2darray r_odd,
                                   const dbl_2darray r_even,
                                   const bool log_sections) : m_alph_even(std::move(alph_even)),
                                                              m_alph_odd(std::move(alph_odd)),
                                                              m_r_even(std::move(r_even)),
                                                              m_r_odd(std::move(r_odd)),
                                                              n(nb_steps), n_done(0),
                                                              b_log(log_sections),
                                                              n_even(r_even.shape(1)),
                                                              n_odd(r_odd.shape(1)),
                                                              s(std::move(s_in)),
                                                              s_even(s.data(), n_even),
                                                              s_odd(s.data() + n_even, n_even),
                                                              f_odd(n_odd), b_even(n_odd),
                                                              Psi_even(n_even), Psi_odd(n_odd),
                                                              p_sections(log_sections ? nb_steps * 2 * (n_even + n_odd) : 0)
    {
    }

    pressure_pair step(const unsigned int i, const double f1, const double bK) override
    {
        // get the current vocal tract attenuation and reflection coefficients
        Eigen::Map<Eigen::ArrayXd> alph_even(DATA_PTR(m_alph_even, i), n_odd);
        Eigen::Map<Eigen::ArrayXd> alph_odd(DATA_PTR(m_alph_odd, i), n_odd);
        Eigen::Map<Eigen::ArrayXd> r_even(DATA_PTR(m_r_even, i), n_even);
        Eigen::Map<Eigen::ArrayXd> r_odd(DATA_PTR(m_r_odd, i), n_odd);

        const bool log = b_log && i < n;
        const unsigned nsec = (n_even + n_odd);
        const unsigned nblk = 2 * nsec;

        // ---even junctions [(F2,B3),(F4,B5),...,(FK-2,BK-1)]->[(F3,B2),(F5,B4),...]--- */
        Psi_even = (s_even - s_odd) * r_even;
        f_odd(0) = f1;
        f_odd(Eigen::seqN(1, n_even)) = s_even + Psi_even;
        b_even(Eigen::seqN(0, n_even)) = s_odd + Psi_even;
        b_even(n_even) = bK;

        if (log)
        {
            // log even-stage forward and backward pressure
            SectionBlock p_section(p_sections.data() + i * nblk, 2, nsec);
            p_section(0, Eigen::seqN(1, n_even, 2)) = s_even;
            p_section(1, Eigen::seqN(1, n_even, 2)) = b_even(Eigen::seqN(0, n_even));
        }

        // propagating over a tube segment
        f_odd *= alph_odd;
        b_even *= alph_even;

        // ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */
        Psi_odd = (f_odd - b_even) * r_odd; //[F3, F5, ...] - [ B2, B4, ... ]
        f_odd += Psi_odd;
        b_even += Psi_odd;

        if (log)
        {
            // log odd-stage output forward pressure
            SectionBlock p_section(p_sections.data() + i * nblk, 2, nsec);
            p_section(0, Eigen::seqN(0, n_odd, 2)) = f_odd;
        }

        // propagating over the other tube segment --- attenuate pressure to "pre-"propagate signal to the output of the section
        f_odd *= alph_even;
        b_even *= alph_odd;

        if (log)
        {
            // log odd-stage output backward pressure
            SectionBlock p_section(p_sections.data() + i * nblk, 2, nsec);
            p_section(1, Eigen::seqN(0, n_odd, 2)) = b_even;

            n_done = i + 1;
        }

        s_even = f_odd(Eigen::seqN(0, n_even));
        s_odd = b_even(Eigen::seqN(1, n_even));

        return pressure_pair(f_odd(n_even), b_even(0));
    }

    nb::ndarray<double, nb::ndim<1>, nb::numpy> get_sout()
    {
        return nb::ndarray<double, nb::ndim<1>, nb::numpy>(s.data(), {2 * n_even});
    }

    nb::ndarray<double, nb::ndim<3>, nb::numpy> get_pout_sections()
    {
        return nb::ndarray<double, nb::ndim<3>, nb::numpy>(p_sections.data(), {n_done, 2, n_even + n_odd});
    }
};

void bind_vocaltract(nb::module_ &m)
{
    nb::class_<WaveReflectionVocalTractRunner, RunnerBase>(m, "WaveReflectionVocalTractRunner")
        .def(nb::init<const unsigned, const dbl_1darray, const dbl_2darray,
                      const dbl_2darray, const dbl_2darray, const dbl_2darray, const bool>(),
             "nb_steps"_a, "sin"_a, "alph_even"_a, "alph_odd"_a, "r_even"_a, "r_odd"_a, "log_sections"_a)
        .def("step", &WaveReflectionVocalTractRunner::step, "i"_a, "f_in"_a, "b_in"_a)
        .def_ro("alph_odd", &WaveReflectionVocalTractRunner::m_alph_odd, nb::rv_policy::reference_internal)
        .def_ro("alph_even", &WaveReflectionVocalTractRunner::m_alph_even, nb::rv_policy::reference_internal)
        .def_ro("r_odd", &WaveReflectionVocalTractRunner::m_r_odd, nb::rv_policy::reference_internal)
        .def_ro("r_even", &WaveReflectionVocalTractRunner::m_r_even, nb::rv_policy::reference_internal)
        .def_prop_ro("sout", &WaveReflectionVocalTractRunner::get_sout, nb::rv_policy::reference_internal)
        .def_ro("n", &WaveReflectionVocalTractRunner::n_done, nb::rv_policy::reference_internal)
        .def_prop_ro("p_sections", &WaveReflectionVocalTractRunner::get_pout_sections, nb::rv_policy::reference_internal);
}
