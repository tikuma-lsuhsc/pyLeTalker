"""`~letalker.typing` Module

The :py:mod:`~typing` module defines the custom type hints.

.. py:currentmodule:: letalker.typing

.. autoclass:: SimResultsDict

.. py:attribute:: TwoLetterVowelLiteral

   Possible values are:


   .. code::

      'aa', 'ii', 'uu', 'ae', 'ih', 'eh', 'ah', 'aw', 'uh', 'oo', 'trach'

   These codes specify vocal tract areas. These can be used for the :code:`areas` 
   argument of the :py:class:`letalker.LeTalkerVocalTract` constructor. Also, 
   they are the keys of the :py:attr:`letalker.constants.vocaltract_areas` 
   dict.
   
   The two-letter vowel codes specify the built-in (supraglottal) vocal tract 
   configurations to produce the specified vowel (44-segment models) while the 
   last option, :code:`"trach"`, is a 32-segment trachea (subglottal) vocal 
   tract model.

.. py:attribute:: StepTypeLiteral

   This literal specifies the transition types supported by 
   :py:class:`letalker.StepGenerator` and other function generators with 
   :code:`transition_type` argument. There are 9 options:

   =================  =========================
   Option             Transition Description
   =================  =========================
   `"step"`           Instantaneous
   `"linear"`         Linear
   `"raised_cos"`     Raised cosine
   `"exp_decay"`      Exponential decay
   `"exp_decay_rev"`  Reverse exponential decay
   `"logistic"`       Logistic function
   `"atan"`           Arctangent function
   `"tanh"`           Hyperbolic tangent
   `"erf"`            Err Function
   =================  =========================

.. attribute:: WhiteNoiseDistributionLiteral

   This literal specifies the supported probability distributions for generating
   noise samples via `WhiteNoiseGenerator` or `ColoredNoiseGenerator`. Possible
   values are:

   =================  ==================================
   Option             Distribution Description
   =================  ==================================
   `"gaussian"`       Gaussian (normal) distribution
   `"two_point"`      Symmetrical two-point distribution
   `"uniform"`        Symmetrical uniform distribution
   =================  ==================================
   
"""

from .sim import SimResultsDict
from .constants import TwoLetterVowelLiteral
from .function_generators.StepGenerator import StepTypeLiteral
from .function_generators.WhiteNoiseGenerator import WhiteNoiseDistributionLiteral

from typing import Literal

TestLiteral = Literal['test','test1'] 


__all__ = [
    "SimResultsDict",
    "TwoLetterVowelLiteral",
    "StepTypeLiteral",
    "WhiteNoiseDistributionLiteral",
]
