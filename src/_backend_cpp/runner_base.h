#pragma once

#include <utility>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>

namespace nb = nanobind;

typedef std::pair<double, double> pressure_pair;

template<typename ...Args3>
using dbl_ndarray = nb::ndarray<double, nb::c_contig, Args3...>;

typedef dbl_ndarray<nb::ndim<1>> dbl_1darray;
typedef dbl_ndarray<nb::ndim<2>> dbl_2darray;
typedef dbl_ndarray<nb::ndim<3>> dbl_3darray;

struct RunnerBase
{
    virtual ~RunnerBase() = default;
    virtual pressure_pair step(const unsigned int i, const double fin, const double bin) = 0;
};

struct FlowNoiseRunnerBase
{
    virtual ~FlowNoiseRunnerBase() = default;
    virtual double step(const unsigned int i, const double uin, const std::list<double> geom) = 0;
};
