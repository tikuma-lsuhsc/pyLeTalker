.. py:module:: letalker


Synthesis Basics
================

The top-level module :py:mod:`~letalker` directly houses all the commonly used 
features of pyLeTalker that are essential to run the voice synthesis while more
advanced features remain only in their respective submodules. The following 
lists summarize all the top-level functions and classes.

* **Sampling rate and time vector**

  .. autosummary::
      :nosignatures:

      fs
      ts

* **Simulation Functions**
  
  .. autosummary::
      :nosignatures:

      sim
      sim_kinematic
      sim_vf

* **Synthesis Elements**

  **Vocal Folds classes**

  .. autosummary::
      :nosignatures:
      
      LeTalkerVocalFolds
      KinematicVocalFolds
      VocalFoldsUg
      VocalFoldsAg

  **Vocal Tract classes**

  .. autosummary::
      :nosignatures:

      LeTalkerVocalTract
      LossyCylinderVocalTract

  **Lungs class**

  .. autosummary::
    :nosignatures:

    LeTalkerLungs
    OpenLungs
    NullLungs

  **Lips class**

  .. autosummary::
      :nosignatures:

      LeTalkerLips

* **Function Generators**

  There are three types of generators to produce parameter samples.


  * Standalone Generators

    .. autosummary::
        :nosignatures:

        Constant
        StepGenerator
        LineGenerator
        Interpolator
        PeriodicInterpolator
        ClampedInterpolator
        FlutterGenerator
        SineGenerator
        ModulatedSineGenerator
        RosenbergGenerator

  * Function Modifiers

    .. autosummary::
        :nosignatures:

        ProductGenerator
        SumGenerator
        ExponentialGenerator
        LogGenerator

  * Random Noise Generators

    .. autosummary::
        :nosignatures:

        WhiteNoiseGenerator
        ColoredNoiseGenerator

Sampling Rate and Time Vector
-----------------------------

.. py:attribute:: fs

   System-wide sampling rate in samples per second. Default rate is 44100 S/s

.. autofunction:: ts

Simulation Functions
--------------------

.. autofunction:: sim
.. autofunction:: sim_kinematic
.. autofunction:: sim_vf

Synthesis Elements
------------------

There are 4 voice production element types:

* Vocal Folds (Glottis)
* Vocal Tract (for both subglottal and supraglottal elements)
* Lungs
* Lips

pyleTalker provides at least one built-in class for each type.

Also, an aspiration noise model can be assigned to a vocal folds class
to inject turbulent aspiration noise which is a function of the glottal
flow.

Vocal Folds classes
^^^^^^^^^^^^^^^^^^^

.. autoclass:: LeTalkerVocalFolds
   :members: Results

.. autoclass:: KinematicVocalFolds
   :members: Results

.. autoclass:: VocalFoldsUg
   :members: Results

.. autoclass:: VocalFoldsAg
   :members: Results

Vocal Tract classes
^^^^^^^^^^^^^^^^^^^

.. autoclass:: LeTalkerVocalTract
   :members: Results

.. autoclass:: LossyCylinderVocalTract
   :members: Results

Lungs classes
^^^^^^^^^^^^^

.. autoclass:: LeTalkerLungs
   :inherited-members: Results

.. autoclass:: OpenLungs
   :inherited-members: Results

.. autoclass:: NullLungs
   :inherited-members: Results

Lips class
^^^^^^^^^^

.. autoclass:: LeTalkerLips
   :members: Results

Aspiration Noise classes
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LeTalkerAspirationNoise
.. autoclass:: KlattAspirationNoise

Function Generators
-------------------

Standalone Generators
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Constant
.. autoclass:: StepGenerator
.. autoclass:: LineGenerator
.. autoclass:: Interpolator
.. autoclass:: PeriodicInterpolator
.. autoclass:: ClampedInterpolator
.. autoclass:: FlutterGenerator
.. autoclass:: SineGenerator
.. autoclass:: ModulatedSineGenerator
.. autoclass:: RosenbergGenerator

Function Modifiers
^^^^^^^^^^^^^^^^^^

.. autoclass:: ProductGenerator
.. autoclass:: SumGenerator
.. autoclass:: ExponentialGenerator
.. autoclass:: LogGenerator

Random Noise Generators
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: WhiteNoiseGenerator
.. autoclass:: ColoredNoiseGenerator
