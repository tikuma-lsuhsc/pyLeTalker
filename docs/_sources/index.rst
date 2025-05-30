pyLeTalker -- Python framework for wave-reflection voice synthesis
==================================================================

|pypi| |pypi-status| |pypi-pyvers| |github-license| 

.. |pypi| image:: https://img.shields.io/pypi/v/pyLeTalker
  :alt: PyPI
.. |pypi-status| image:: https://img.shields.io/pypi/status/pyLeTalker
  :alt: PyPI - Status
.. |pypi-pyvers| image:: https://img.shields.io/pypi/pyversions/pyLeTalker
  :alt: PyPI - Python Version
.. |github-license| image:: https://img.shields.io/github/license/tikuma-lsuhsc/pyLeTalker
  :alt: GitHub License

.. .. |github-status| image:: https://img.shields.io/github/actions/workflow/status/tikuma-lsuhsc/pyLeTalker/test_n_pub.yml?branch=main
..   :alt: GitHub Workflow Status

**pyLeTalker** is a Python library to synthesize (or simulate) human voice production, 
intended for research use. It offers flexibility to test different numerical 
voice folds model and to synthesize various pathologial voice. It is currently 
not suitable to synthesize speech. 

This library started as a repackaged version of `LeTalker, a Matlab GUI demo by 
Dr. Brad Story <https://sites.arizona.edu/sapl/research/le-talker/>`_, and it 
since has been evolved as a flexible general voice synthesis library built 
around the wave-reflection vocal tract model. The wave-reflection vocal tract 
model treats the vibrating air pressure as propagating waves through the vocal 
tract. Some portion of the incidental wave reflects when it encounters a change 
in the cross-sectional area of the vocal tract.

.. image:: images/wave-reflection-model-light.png
    :class: only-light

.. image:: images/wave-reflection-model-dark.png
    :class: only-dark

Please note that pyLeTalker will be perpetually in active development. 
As such, *feedback* and possibly *contributions* are highly appreciated.
Please read the :ref:`use-contribute` section below.

.. Drop by our `Gitter chat room <https://gitter.im/PraatParselmouth/Lobby>`_ or post a message to our `Google discussion group <https://groups.google.com/d/forum/parselmouth>`_ if you have any question, remarks, or requests!

.. .. only:: html

..     .. note::

..         Try out Parselmouth online, in interactive Jupyter notebooks on Binder: |binder_badge_examples|


Installation
------------

Install the :py:mod:`pyLeTalker` package via :code:`pip`.

.. code-block::

   pip install pyLeTalker

Simple Examples
---------------

The following closely reproduces the `LeTalker demo <https://sites.arizona.edu/sapl/research/le-talker/>`_'s 
default configuration:

.. code-block:: python

   import letalker as lt

   fs = lt.fs # sampling rate (44.1 kHz)

   T = 0.5 # simulation duration
   N = round(T*lt.fs) # number of samples to generate

   # define the simulation elements using the default parameter values
   vf = lt.LeTalkerVocalFolds() # 3-mass vocal fold model with muscle activation
   vt = lt.LeTalkerVocalTract('aa') # supraglottal vocal tract to produce /a/
   tr = lt.LeTalkerVocalTract('trach') # trachea / subglottal tract
   lp = lt.LeTalkerLips() # Ishizaka-Flanagan lip model
   lg = lt.LeTalkerLungs() # constant lung pressure 

   # run the simulation
   pout, res = lt.sim(N, vf, vt, tr, lg, lp) 
   # pout - radiated sound pressure signal (a 1D NumPy Array)
   # res - a dict of Results objects of all the simulation elements

The above script synthesizes an /a/ vowel sustained for 0.5 second of phonating /a/
approximately at :math:`f_o\approx` 100 Hz. `lt.sim()` is the basic simulation function,
and it is equipped with the default arguments so the above example can be shortened
to  

.. code-block:: python

   # only need to define the vocal folds element
   vf = lt.LeTalkerVocalFolds() # 3-mass vocal fold model with muscle activation

   # run the simulation with default subglottal tract, lips, and lungs elements as
   # well as the supraglottal tract with the specified vowel (/a/ in this case)
   pout, res = lt.sim(N, vf, 'aa') 

The kinematic vocal fold model (:py:class:`~KinematicVocalFolds`) is a potent vocal fold 
model to provide more predictable outcomes, especially $f_o$, by trading off the
interaction between the glottal flow and glottal area for programmable vocal fold
geometry and motion. 

.. code-block:: python

   fo = 100 # fundamental frequency in Hz

   # define the simulation elements 
   vf = lt.KinematicVocalFolds(fo) # 3-mass vocal fold model with muscle activation

   # run the simulation
   pout, res = lt.sim(N, vf, 'aa') 

The kinematic vocal folds get their own simulation function for convenience:

.. code-block:: python

   pout, res = lt.sim_kinematic(N, fo, 'aa') 

.. _use-contribute:

Use and Contribution
--------------------

The development and documentation of pyLeTalker is heavily biased towards the
current research interests of the active developers. This project is offered as 
open-source to attract other voice researchers to get involved even just to use 
the library. 

If you are interested in using pyLeTalker but unsure of how to use 
it, please post your interest on [the Discussion board](https://github.com/tikuma-lsuhsc/pyLeTalker/discussions)
or email me.
I will help you setting up your script. This will help refining the available-but-
not-yet-used features (both debugging and documentation) or adding new features
that are not currently included. 

If you are a Python programmer/researcher and encountered a bug or needing a new 
feature, please post on [the Issues board](https://github.com/tikuma-lsuhsc/pyLeTalker/issues),
and I'll be happy to discuss the prospect. I will accept [pull requests](https://github.com/tikuma-lsuhsc/pyLeTalker/pulls)
but only after reviewing their values and qualities.


.. toctree::
    :maxdepth: 2
    :caption: Contents

    examples
    citations
    api_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
