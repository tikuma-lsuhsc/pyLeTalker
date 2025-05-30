.. py:currentmodule:: letalker

Citation Guidelines
===================

Citing ``pyLeTalker``
---------------------

To cite ``pyLeTalker`` as a software:

    Ikuma, T. (2025). pyLeTalker: wave-reflection voice synthesis framework [Computer program]. Version |version|, https://github.com/tikuma-lsuhsc/pyLeTalker

.. parsed-literal::

    @misc{pyletalker,
    author = {Takeshi Ikuma},
    title = {pyLeTalker: wave-reflection voice synthesis framework (v.|version|)},
    year  = {2025},
    url   = {\url{https://github.com/tikuma-lsuhsc/pyLeTalker}},
    }

There is a manuscript introducing ``pyLeTalker`` currently under preparation.

Citing computational models used in ``pyLeTalker``
--------------------------------------------------

The various numerical models that are included in this library were developed 
mostly by other researchers. It is more important for academic studies to cite
their papers as ``pyLeTalker`` is merely a tool. Please use the list below as a
guideline to choose appropriate papers for the ``pyLeTatker`` element classes
that you employed.

:py:class:`LeTalkerVocalTract` (the default vocal tract model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   liljencrants1985
   story1995a
   story1996

Note
^^^^

* Cite :cite:t:`story1996` only if the default vocal tract cross-sectional areas 
  are used (i.e., specified the default vowels).

:py:class:`LeTalkerVocalFolds`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   story1995
   titze2002
   titze2002a


:py:class:`KinematicVocalFolds` or :py:func:`sim_kinematic`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   titze1984
   titze1989a
   titze1989b

:py:class:`VocalFoldsUg` & :py:class:`VocalFoldsAg`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   titze1984

:py:class:`LeTalkerAspirationNoise`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   samlan2011

Note
""""

This aspiration noise model automatically used if a vocal-fold model is simulated 
with ``aspiration_noise=True``.

:py:class:`KlattAspirationNoise`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   klatt1990

:py:class:`FlutterGenerator`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   klatt1990

Use of :py:class:`ModulatedSineGenerator` for subharmonic vocal fold modulation with :py:class:`KinematicVocalFolds`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. bibliography:: 
   :filter: False
    
   ikuma2025a
