"""
References:
[1] B. H. Story and I. R. Titze, “Voice simulation with a body‐cover model of the vocal folds,” The Journal of the Acoustical Society of America, vol. 97, no. 2, pp. 1249–1260, Feb. 1995, doi: 10.1121/1.412234
[1] B. H. Story, I. R. Titze, and E. A. Hoffman, “Vocal tract area functions from magnetic resonance imaging,” The Journal of the Acoustical Society of America, vol. 100, no. 1, pp. 537–554, Jul. 1996, doi: 10.1121/1.415960
[1] I. R. Titze, “Parameterization of the glottal area, glottal flow, and vocal fold contact area,” J. Acoust. Soc. Am., vol. 75, no. 2, pp. 570–580, 1984, doi: 10.1121/1.390530
[1] I. R. Titze, “The physics of small-amplitude oscillation of the vocal folds,” J. Acoust. Soc. Am., vol. 83, no. 4, pp. 1536–1552, 00 1988, doi: 10.1121/1.395910
[1] I. R. Titze, Workshop on Acoustic Voice Analysis: Summary Statement. Denver, CO, USA: National Center for Voice and Speech, 1994. Available: https://ncvs.org/archive/freebooks/summary-statement.pdf. [Accessed: Jul. 11, 2022]
[1] I. Titze, T. Riede, and T. Mau, “Predicting achievable fundamental frequency ranges in vocalization across species,” PLoS Comput Biol, vol. 12, no. 6, p. e1004907, Jun. 2016, doi: 10.1371/journal.pcbi.1004907
[1] I. R. Titze, “A four-parameter model of the glottis and vocal fold contact area,” Speech Communication, vol. 8, no. 3, pp. 191–201, Sep. 1989, doi: 10.1016/0167-6393(89)90001-0


"""

from __future__ import annotations

from typing import Literal, TypedDict
from collections.abc import Callable
from numpy.typing import ArrayLike, NDArray
from numbers import Number

from .elements.abc import (
    VocalFolds,
    VocalTract,
    Lips,
    Lungs,
    AspirationNoise,
    Element,
    BlockRunner,
)
from .function_generators.abc import FunctionGenerator, AnalyticFunctionGenerator

from math import pi
import numpy as np

from .core import compile_njit_if_numba
from .elements import (
    LeTalkerVocalTract,
    LeTalkerLips,
    LeTalkerLungs,
    KinematicVocalFolds,
    NullLungs,
)
from .function_generators import SineGenerator, Constant, ClampedInterpolator
from .constants import vocaltract_areas, PL


class SimResultsDict(TypedDict):
    """Dict of letalker simulation per-element outputs"""

    lungs: Element.Results
    trachea: Element.Results
    vocalfolds: Element.Results
    vocaltracts: Element.Results
    lips: Element.Results


def _sim_init(
    vocalfolds: VocalFolds,
    vocaltract: VocalTract | ArrayLike | str | dict,
    trachea: VocalTract | ArrayLike | dict | None,
    lungs: Lungs | float | None,
    lips: Lips | dict | None,
    aspiration_noise: AspirationNoise | bool | dict | None,
) -> tuple[VocalFolds, VocalTract, VocalTract, Lungs, Lips]:

    if not isinstance(vocaltract, VocalTract):
        vocaltract = (
            LeTalkerVocalTract(**vocaltract)
            if isinstance(vocaltract, dict)
            else LeTalkerVocalTract(vocaltract)
        )

    if not isinstance(trachea, VocalTract):
        if isinstance(trachea, dict) and "area" not in trachea:
            trachea["areas"] = "trach"
        trachea = (
            LeTalkerVocalTract(**trachea)
            if isinstance(trachea, dict)
            else LeTalkerVocalTract(trachea or "trach")
        )

    if not isinstance(lungs, Lungs):
        lungs = LeTalkerLungs(lungs)

    if not isinstance(lips, Lips):
        lips = LeTalkerLips() if lips is None else LeTalkerLips(**lips)

    if aspiration_noise is not None:
        vocalfolds.add_aspiration_noise(aspiration_noise)

    # link vocal tracts to neighboring elements
    lips.link(vocaltract)
    vocalfolds.link(trachea, vocaltract)
    lungs.link(trachea)

    return lungs, trachea, vocalfolds, vocaltract, lips


def _sim_loop(
    nb_steps: int,
    lungs: BlockRunner,
    trachea: BlockRunner,
    vf: BlockRunner,
    vt: BlockRunner,
    lips: BlockRunner,
):

    flung = blung = fsg = bsg = feplx = beplx = flip = blip = 0.0

    for i in range(nb_steps):
        # Compute current pressure outputs from VOCAL FOLD
        feplx, bsg = vf.step(i, fsg, beplx)

        # Compute the next states of flip & beplx
        flip, beplx = vt.step(i, feplx, blip)
        blip = lips.step(i, flip)

        # Compute the next states of fsg & blung
        flung = lungs.step(i, blung)
        fsg, blung = trachea.step(i, flung, bsg)


def sim(
    nb_samples: int,
    vocalfolds: VocalFolds,
    vocaltract: str | ArrayLike | FunctionGenerator | dict | VocalTract,
    trachea: ArrayLike | FunctionGenerator | dict | VocalTract | None = None,
    lungs: float | FunctionGenerator | Lungs | None = None,
    lips: Lips | None = None,
    aspiration_noise: bool | dict | AspirationNoise | None = None,
    *,
    return_results: bool = True,
    n0: int = 0,
) -> NDArray | tuple[NDArray, SimResultsDict]:
    """Run simulation

    Parameters
    ----------
    nb_samples
        Number of samples to generate
    vocalfolds
        Vocal fold model
    vocaltract
        Supraglottal vocal tract. If a 2-letter vowel name (e.g., `'aa'` & `'ii'`) is passed,
        the default `LeTalkerVocalTract` model will be used with a male vocal tract, configured
        to the specified vowel. If 1D or 2D array or a `FunctionGenerator` object is passed,
        it will be used as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of the `LeTalkerVocalTract`,
        you may pass in a dict of key-value pairs. A custom `VocalTract` model object could also
        be passed.
    trachea, optional
        Subglottal vocal tract model, by default None to use the `LeTalkerVocalTract`
        model with the default trachea cross-sectional areas (`'trach'`). If 1D
        or 2D array or a `FunctionGenerator` object is passed, it will be used
        as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of
        the `LeTalkerVocalTract`, you may pass in a dict of key-value pairs.
        A custom `VocalTract` model object could also be passed.
    lungs, optional
        Lungs model, by default None to use `LeTalkerLungs` model with the default 7840 dyn/cm² lung
        pressure level (with the onset time of 2 ms). A float value changes the pressure level while
        a FunctionGenerator object can be specified to alter the temporal profile. A custom `Lungs` model
        object could also be passed.
    lips, optional
        Lips model, by default None to use `LeTalkerLips` model (Ishizaka-Flanagan lips radiation model). A
        custom `Lips` model object could also be passed.
    aspiration_noise, optional
        Aspiration noise model, by default None to disable aspiration noise injection (or keep the default
        aspiration noise mode of the specified `VocalFolds` model). Passing in `True` enables the injection
        model (LeTalkerAspirationNoise) at the default level and threshold. Use a key-value dict to customize
        LeTalkerAspirationNoise further or pass in a custom  `AspirationNoise` model object.
    return_results, optional
        If `True`, also return the results dict of all the simulation elements, by default True
    n0, optional
        Simulation starting time in sample index, by default 0

    Returns
    -------
        pout :  NDArray
            Radiated acoustic pressure signal
        results : dict[SIM_RESULTS_KEY, Element.Results], optional
            The parameters and results of the simulation elements
    """

    components = _sim_init(
        vocalfolds, vocaltract, trachea, lungs, lips, aspiration_noise
    )
    # -> lungs, trachea, vocalfolds, vocaltract, lips

    runners = [c.create_runner(nb_samples, n0) for c in components]

    compile_njit_if_numba(_sim_loop, fastmath=True)(nb_samples, *runners)

    # extract the acoustic output
    lips = components[-1]
    pout = runners[-1].pout

    if not return_results:
        return pout

    return pout, {
        name: c.create_result(r, n0=n0)
        for name, c, r in zip(
            ("lungs", "trachea", "vocalfolds", "vocaltract", "lips"),
            components,
            runners,
        )
    }


def sim_kinematic(
    nb_samples: int,
    xi_ref: float | AnalyticFunctionGenerator,
    vocaltract: str | ArrayLike | FunctionGenerator | dict | VocalTract,
    trachea: ArrayLike | FunctionGenerator | dict | VocalTract | None = None,
    lungs: float | FunctionGenerator | Lungs | None = None,
    lips: Lips | None = None,
    aspiration_noise: bool | dict | AspirationNoise | None = None,
    *,
    return_results: bool = True,
    n0: int = 0,
    **kvf_kws,
) -> NDArray | tuple[NDArray, SimResultsDict]:
    """Run simulation with kinematic vocal fold model

    Parameters
    ----------
    nb_samples
        Number of samples to generate
    xi_ref
        reference vocal fold motion (periodicity/qusiperiodicity required to )
    L0
        relaxed vocal fold length in cm
    T0
        relaxed vocal fold thickness in cm
    fo2L
        fo-to-length conversion, or 2 coefficients of Titze-Riede-Mau conversion
    xim
        vibration maximum amplitude in cm
    vocaltract
        Supraglottal vocal tract. If a 2-letter vowel name (e.g., 'aa' & 'ii') is passed,
        the default `LeTalkerVocalTract` model will be used with a male vocal tract, configured
        to the specified vowel. If 1D or 2D array or a `FunctionGenerator` object is passed,
        it will be used as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of the `LeTalkerVocalTract`,
        you may pass in a dict of key-value pairs. A custom `VocalTract` model object could also
        be passed.
    trachea, optional
        Subglottal vocal tract model, by default None to use the `LeTalkerVocalTract`
        model with the default trachea cross-sectional areas (`'trach'`). If 1D
        or 2D array or a `FunctionGenerator` object is passed, it will be used
        as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of
        the `LeTalkerVocalTract`, you may pass in a dict of key-value pairs.
        A custom `VocalTract` model object could also be passed.
    lungs, optional
        Lungs model, by default None to use `LeTalkerLungs` model with the default 7840 dyn/cm² lung
        pressure level (with the onset time of 2 ms). A float value changes the pressure level while
        a FunctionGenerator object can be specified to alter the temporal profile. A custom `Lungs` model
        object could also be passed.
    lips, optional
        Lips model, by default None to use `LeTalkerLips` model (Ishizaka-Flanagan lips radiation model). A
        custom `Lips` model object could also be passed.
    aspiration_noise, optional
        Aspiration noise model, by default None to disable aspiration noise injection (or keep the default
        aspiration noise mode of the specified `VocalFolds` model). Passing in `True` enables the injection
        model (LeTalkerAspirationNoise) at the default level and threshold. Use a key-value dict to customize
        LeTalkerAspirationNoise further or pass in a custom  `AspirationNoise` model object.
    return_results, optional
        If `True`, also return the results dict of all the simulation elements, by default True
    n0, optional
        Simulation starting time in sample index, by default 0

    Returns
    -------
        pout :  NDArray
            Radiated acoustic pressure signal
        results : dict[SIM_RESULTS_KEY, Element.Results], optional
            The parameters and results of the simulation elements
    """
    # complex-valued reference edge displacement waveform
    # relaxed vocal fold length in cm (1.6 for male, 1.0 for female)
    # relaxed vocal fold thickness
    # fo to vf length converter

    vocalfolds = KinematicVocalFolds(xi_ref, **kvf_kws)

    return sim(
        nb_samples,
        vocalfolds,
        vocaltract,
        trachea,
        lungs,
        lips,
        aspiration_noise,
        return_results=return_results,
        n0=n0,
    )


def sim_vf(
    nb_samples: int,
    vocalfolds: VocalFolds,
    *,
    Asub: float | FunctionGenerator | Literal[False] | None = None,
    Asup: float | FunctionGenerator | Literal[False] | None = None,
    Fin: float | ArrayLike | FunctionGenerator | None = None,
    Bin: float | ArrayLike | FunctionGenerator | None = None,
    n0: int = 0,
) -> Element.Results:
    """Run simulation only with a vocal fold model

    The given vocal fold model is simulated with the optional subglottal and
    supraglottal conditions. Each condition is set by the interfacing cross-sectional
    area (Asub/Asup) and the input pressure function (forward pressure Fin for
    subglottal input/backward pressure Bin for supraglottal input).

    Parameters
    ----------
    nb_samples
        Number of samples to generate
    vocalfolds
        Vocal fold model
    Asub, optional
        Subglottal area, by default None to use the default subglottal area
        (i.e., the last element of `constants.vocaltract_area['trach']`, 0.31 cm²)
    Asup, optional
        Epiglottal area, by default None to use the epiglottal area of default
        /a/ vocal tract (i.e., the first element of `constants.vocaltract_area['aa']`,
        0.56 cm²)
    Fin, optional
        Subglottal input pressure, by default None to use `constants.PL` (7840 dyn/cm²)
    Bin, optional
        Epiglottal input pressure, by default None to use 0
    n0, optional
        starting time sample offset, by default 0

    Returns
    -------
        The result object of the specified `vocaltracts`.
    """

    if Asub is None and vocalfolds.upstream is None:
        Asub = vocaltract_areas["trach"][-1]
    if Asup is None and vocalfolds.downstream is None:
        Asup = vocaltract_areas["aa"][0]

    if Asub is not None or Asup is not None:
        vocalfolds.link(Asub, Asup)

    def set_pressure(P, P_default, name):
        if P is None:
            P = P_default

        if isinstance(P, FunctionGenerator):
            return P

        try:
            P = np.asarray(P, dtype=float)
            assert P.ndim <= 1
        except (ValueError, AssertionError) as e:
            raise ValueError(
                f"{name} must be either float or a numerical 1D array."
            ) from e

        return (
            Constant(P.reshape(()))
            if P.size == 1
            else ClampedInterpolator(vocalfolds.fs, P)
        )

    Fin = set_pressure(Fin, PL, "Fin")
    Bin = set_pressure(Bin, 0.0, "Bin")

    # create their simulation runner objects
    runner = vocalfolds.create_runner(nb_samples, n0)

    f_in = Fin(nb_samples, n0, force_time_axis="tile_data")
    b_in = Bin(nb_samples, n0, force_time_axis="tile_data")

    for i, (fi, bi) in enumerate(zip(f_in, b_in)):
        # Compute the next states of bout
        runner.step(i, fi, bi)

    return vocalfolds.create_result(runner)


###################


def _sim_sep_loop(
    nb_steps: int,
    h_lung: BlockRunner,
    h_trachea: BlockRunner,
    m_vf: BlockRunner,
    h_vt: BlockRunner,
    h_lip: BlockRunner,
    n_lung: BlockRunner,
    n_trachea: BlockRunner,
    a_vf: BlockRunner,
    n_vt: BlockRunner,
    n_lip: BlockRunner,
):

    hblung = hfsg = hbeplx = hblip = 0.0  # harmonic component
    nblung = nfsg = nbeplx = nblip = 0.0  # nonharmonic component
    for i in range(nb_steps):
        # Compute harmonics+noise vf
        feplx, bsg = m_vf.step(i, hfsg + nfsg, hbeplx + nbeplx)

        # Compute harmonics-only vf
        hfeplx, hbsg = a_vf.step(i, hfsg, hbeplx)

        # compute the noise components of the vf outputs
        nfeplx = feplx - hfeplx
        nbsg = bsg - hbsg

        # Compute the next states of flip & beplx
        # - harmonic branch
        hflip, hbeplx = h_vt.step(i, hfeplx, hblip)
        hblip = h_lip.step(i, hflip)

        # - noise branch
        nflip, nbeplx = n_vt.step(i, nfeplx, nblip)
        nblip = n_lip.step(i, nflip)

        # Compute the next states of fsg & blung
        # - harmonic branch
        hflung = h_lung.step(i, hblung)
        hfsg, hblung = h_trachea.step(i, hflung, hbsg)

        # - noise branch (zero input)
        nflung = n_lung.step(i, nblung)
        nfsg, nblung = n_trachea.step(i, nflung, nbsg)


def _sim_dual(
    nb_samples: int,
    vocalfolds: VocalFolds,
    vocaltract: VocalTract | ArrayLike | str,
    trachea: VocalTract | ArrayLike | None = None,
    lungs: Lungs | float | None = None,
    lips: Lips | None = None,
    aspiration_noise: AspirationNoise | None = None,
    n0: int = 0,
) -> tuple[
    Element.Results, Element.Results, Element.Results, Element.Results, Element.Results
]:
    """Run harmonic/noise dual-path simulation

    Parameters
    ----------
    nb_samples
        Number of samples to generate
    vocalfolds
        Vocal fold model
    vocaltract
        Supraglottal vocal tract. If a 2-letter vowel name (e.g., 'aa' & 'ii') is passed,
        the default `LeTalkerVocalTract` model will be used with a male vocal tract, configured
        to the specified vowel. If 1D or 2D array or a `FunctionGenerator` object is passed,
        it will be used as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of the `LeTalkerVocalTract`,
        you may pass in a dict of key-value pairs. A custom `VocalTract` model object could also
        be passed.
    trachea, optional
        Subglottal vocal tract model, by default None to use the default trachea cross-sectional
        areas with the `LeTalkerVocalTract` model. If 1D or 2D array or a `FunctionGenerator` object is
        passed, it will be used as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of the `LeTalkerVocalTract`,
        you may pass in a dict of key-value pairs. A custom `VocalTract` model object could also
        be passed.
    lungs, optional
        Lungs model, by default None to use `LeTalkerLungs` model with the default 7840 dyn/cm² lung
        pressure level (with the onset time of 2 ms). A float value changes the pressure level while
        a FunctionGenerator object can be specified to alter the temporal profile. A custom `Lungs` model
        object could also be passed.
    lips, optional
        Lips model, by default None to use `LeTalkerLips` model (Ishizaka-Flanagan lips radiation model). A
        custom `Lips` model object could also be passed.
    aspiration_noise, optional
        Aspiration noise model, by default None to disable aspiration noise injection (or keep the default
        aspiration noise mode of the specified `VocalFolds` model). Passing in `True` enables the injection
        model (LeTalkerAspirationNoise) at the default level and threshold. Use a key-value dict to customize
        LeTalkerAspirationNoise further or pass in a custom  `AspirationNoise` model object.
    return_results, optional
        If `True`, also return the results dict of all the simulation elements, by default True
    n0, optional
        Simulation starting time in sample index, by default 0

    Returns
    -------
        lip_main_result :  Element.Results
            harmonics-only lips results
        vocal_fold_main_result :  Element.Results
            full vocal folds results
        lip_aux_result :  Element.Results
            non-harmonics-only lips results
        vocal_fold_aux_result :  Element.Results
            harmonics-only vocal folds results
        lung_main_result :  Element.Results
            lung results

    Overview
    --------

    Typically, only vocal folds models are nonlinear while the rest of the simulation elements are linear. To
     harmonics and non-harmonic components

    """

    lung_null = NullLungs()

    components = _sim_init(
        vocalfolds, vocaltract, trachea, lungs, lips, aspiration_noise
    )
    # -> lungs, trachea, vocalfolds, vocaltract, lips

    main_runners = [c.create_runner(nb_samples, n0) for c in components]

    aux_runners = [
        lung_null.create_runner(nb_samples, n0),
        *(
            (
                c.create_runner(nb_samples, n0, noise_free=True)
                if isinstance(c, VocalFolds)
                else c.create_runner(nb_samples, n0)
            )
            for c in components[1:]
        ),
    ]

    compile_njit_if_numba(_sim_sep_loop, fastmath=True)(
        nb_samples, *main_runners, *aux_runners
    )

    # extract the acoustic output
    main_res = (components[i].create_result(main_runners[i], n0=n0) for i in (4, 2))
    aux_res = (components[i].create_result(aux_runners[i], n0=n0) for i in (4, 2))

    return *main_res, *aux_res, components[0].create_result(main_runners[0], n0=n0)


def _sim_dual_kinematic(
    nb_samples: int,
    xi_ref: (
        float | AnalyticFunctionGenerator
    ),  # complex-valued reference edge displacement waveform
    L0: float,  # relaxed vocal fold length in cm (1.6 for male, 1.0 for female)
    T0: float,  # relaxed vocal fold thickness
    fo2L: tuple[float, float] | Callable,  # fo to vf length converter
    xim: float,
    vocaltract: VocalTract | ArrayLike | str,
    trachea: VocalTract | ArrayLike | None = None,
    lungs: Lungs | float | None = None,
    lips: Lips | None = None,
    aspiration_noise: AspirationNoise | None = None,
    n0: int = 0,
    **kvf_kws,
) -> tuple[
    Element.Results, Element.Results, Element.Results, Element.Results, Element.Results
]:
    """Run harmonic/noise dual-path simulation with kinematic vocal fold model

    Parameters
    ----------
    nb_samples
        Number of samples to generate
    xi_ref
        reference vocal fold motion (periodicity/qusiperiodicity required to )
    L0
        relaxed vocal fold length in cm
    T0
        relaxed vocal fold thickness in cm
    fo2L
        fo-to-length conversion, or 2 coefficients of Titze-Riede-Mau conversion
    xim
        vibration maximum amplitude in cm
    vocaltract
        Supraglottal vocal tract. If a 2-letter vowel name (e.g., 'aa' & 'ii') is passed,
        the default `LeTalkerVocalTract` model will be used with a male vocal tract, configured
        to the specified vowel. If 1D or 2D array or a `FunctionGenerator` object is passed,
        it will be used as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of the `LeTalkerVocalTract`,
        you may pass in a dict of key-value pairs. A custom `VocalTract` model object could also
        be passed.
    trachea, optional
        Subglottal vocal tract model, by default None to use the default trachea cross-sectional
        areas with the `LeTalkerVocalTract` model. If 1D or 2D array or a `FunctionGenerator` object is
        passed, it will be used as the cross-sectional areas (in cm²) of vocal tract sections (if 2D,
        the first dimension is the time axes). To specify other parameters of the `LeTalkerVocalTract`,
        you may pass in a dict of key-value pairs. A custom `VocalTract` model object could also
        be passed.
    lungs, optional
        Lungs model, by default None to use `LeTalkerLungs` model with the default 7840 dyn/cm² lung
        pressure level (with the onset time of 2 ms). A float value changes the pressure level while
        a FunctionGenerator object can be specified to alter the temporal profile. A custom `Lungs` model
        object could also be passed.
    lips, optional
        Lips model, by default None to use `LeTalkerLips` model (Ishizaka-Flanagan lip radiation model). A
        custom `Lips` model object could also be passed.
    aspiration_noise, optional
        Aspiration noise model, by default None to disable aspiration noise injection (or keep the default
        aspiration noise mode of the specified `VocalFolds` model). Passing in `True` enables the injection
        model (LeTalkerAspirationNoise) at the default level and threshold. Use a key-value dict to customize
        LeTalkerAspirationNoise further or pass in a custom  `AspirationNoise` model object.
    return_results, optional
        If `True`, also return the results dict of all the simulation elements, by default True
    n0, optional
        Simulation starting time in sample index, by default 0

    Returns
    -------
        lip_main_result :  Element.Results
            harmonics-only lip results
        vocal_fold_main_result :  Element.Results
            full vocal folds results
        lip_aux_result :  Element.Results
            non-harmonics-only lip results
        vocal_fold_aux_result :  Element.Results
            harmonics-only vocal folds results
        lung_main_result :  Element.Results
            lung results
    """

    if isinstance(xi_ref, Number):
        # given fo
        xi_ref = SineGenerator(xi_ref, phi0=-pi / 2)

    if L0 is not None:
        kvf_kws["L0"] = L0
    if T0 is not None:
        kvf_kws["T0"] = T0
    if fo2L is not None:
        kvf_kws["fo2L"] = fo2L
    if xim is not None:
        kvf_kws["xim"] = xim

    vocalfolds = KinematicVocalFolds(xi_ref, **kvf_kws)

    return sim_dual(
        nb_samples,
        vocalfolds,
        vocaltract,
        trachea,
        lungs,
        lips,
        aspiration_noise,
        n0=n0,
    )
