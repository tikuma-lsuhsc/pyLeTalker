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

void bind_flow_noise(nb::module_ &m)
{
    nb::class_<NoFlowNoiseRunner, FlowNoiseRunnerBase>(m, "NoFlowNoiseRunner")
        .def(nb::init<>())
        .def("step", &NoFlowNoiseRunner::step, "i"_a, "u_in"_a, "geom"_a);
}
