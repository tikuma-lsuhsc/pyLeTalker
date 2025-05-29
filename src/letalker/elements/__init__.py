"""

Creating Custom Voice Production Element Classes  
================================================

You can create your own Python classes for both the voice production elements 
and function generators to custom models or parameter behaviors.

The main module :py:mod:`~letalker` provides all the functions and classes that
are essential to run the voice synthesis.


API Reference
-------------
     
.. autosummary::

    abc.Element
    abc.VocalFolds
    abc.VocalTract
    abc.Lungs
    abc.Lips

.. autoclass:: letalker.elements.abc.Element
.. autoclass:: letalker.elements.abc.VocalFolds
.. autoclass:: letalker.elements.abc.VocalTract
.. autoclass:: letalker.elements.abc.Lungs
.. autoclass:: letalker.elements.abc.Lips
"""

from . import abc
from .KinematicVocalFolds import KinematicVocalFolds
from .VocalFoldsUg import VocalFoldsUg
from .VocalFoldsAg import VocalFoldsAg
from .LeTalkerVocalFolds import LeTalkerVocalFolds
from .LeTalkerVocalTract import LeTalkerVocalTract
from .Lungs import LeTalkerLungs, OpenLungs, NullLungs
from .LeTalkerLips import LeTalkerLips
from .LeTalkerAspirationNoise import LeTalkerAspirationNoise, AspirationNoise
from .LossyCylinderVocalTract import LossyCylinderVocalTract
from .KlattAspirationNoise import KlattAspirationNoise
