"""
Creating Custom Function Generator Classes
================================================

You can create your own Python classes for both the voice production elements
and function generators to custom models or parameter behaviors.

The main module :py:mod:`~letalker` provides all the functions and classes that
are essential to run the voice synthesis.

API Reference
-------------

.. autosummary::

    abc.SampleGenerator
    abc.FunctionGenerator
    abc.AnalyticFunctionGenerator
    abc.NoiseGenerator


.. autoclass:: letalker.function_generators.abc.SampleGenerator
.. autoclass:: letalker.function_generators.abc.FunctionGenerator
.. autoclass:: letalker.function_generators.abc.AnalyticFunctionGenerator
.. autoclass:: letalker.function_generators.abc.NoiseGenerator
"""

from . import abc
from .Constant import Constant
from .FlutterGenerator import FlutterGenerator
from .StepGenerator import StepGenerator
from .SineGenerator import SineGenerator
from .ModulatedSineGenerator import ModulatedSineGenerator
from .WhiteNoiseGenerator import WhiteNoiseGenerator
from .ColoredNoiseGenerator import ColoredNoiseGenerator
from .LineGenerator import LineGenerator
from .Interpolator import Interpolator, PeriodicInterpolator, ClampedInterpolator
from .ProductGenerator import ProductGenerator
from .SumGenerator import SumGenerator
from .RosenbergGenerator import RosenbergGenerator
from .ExponentialGenerator import ExponentialGenerator
from .LogGenerator import LogGenerator

__all__ = [
    "abc",
    "Constant",
    "FlutterGenerator",
    "StepGenerator",
    "LineGenerator",
    "Interpolator",
    "PeriodicInterpolator",
    "ClampedInterpolator",
    "SineGenerator",
    "ModulatedSineGenerator",
    "RosenbergGenerator",
    "ColoredNoiseGenerator",
    "WhiteNoiseGenerator",
    "ProductGenerator",
    "SumGenerator",
    "LogGenerator",
    "ExponentialGenerator",
]
