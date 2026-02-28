# from numba.core.errors import NumbaExperimentalFeatureWarning

# warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

from letalker.constants import male_vf_params
from letalker.elements import (
    KinematicVocalFolds,
    LeTalkerLips,
    LeTalkerLungs,
    LeTalkerVocalTract,
)
from letalker.sim import sim, sim_kinematic


def test_sim():

    N = 100  # fs  # 10000  # fs
    fo = 101
    vocaltract = LeTalkerVocalTract("aa")
    trachea = LeTalkerVocalTract("trach")
    lungs = LeTalkerLungs()
    lips = LeTalkerLips()

    vf = KinematicVocalFolds(fo, **male_vf_params)
    # vf.add_aspiration_noise(False)

    po, parts = sim(N, vf, vocaltract, trachea, lungs, lips)

    lips, vf, lungs = (parts[c] for c in ("lips", "vocalfolds", "lungs"))
    ug, psg, pe = vf.ug, vf.psg, vf.peplx

    vf.glottal_area
    vf.contact_area
    vf.length
    vf.y
    vf.z

    if vf.aspiration_noise:
        vf.aspiration_noise.unoise
        vf.aspiration_noise.nu_inv
        vf.aspiration_noise.RE2b


def test_sim_kinematic():

    N = 100

    vf_params = {
        **male_vf_params,
        # "xi0": None,  # prephonatory displacements of the edge of the vf at the level of the vocal process
        # "xim": 0.1,  # common amplitude of the upper and lower edges of the folds
        # "Qa": 0.3,  # abduction quotient: xi02/xim (Titze 84 Eq (8))
        # "Qs": 3.0,  # shape quotient: (xi01-xi02)/xim (Titze 84 Eq (9))
        # "Qp": 0.2,  # phase quotient: phi/2pi (Titze 84 Eq (10))
        # "Qb": 1.0,  # bulging quotient: (|xi01-xi02|/2-(xib-xi02))/xim
    }

    po, parts = sim_kinematic(
        N, 101, **vf_params, vocaltract="aa", aspiration_noise=True
    )

    vf, lungs = (parts[c] for c in ("vocalfolds", "lungs"))

    # make sure
    ug, psg, pe = vf.ug, vf.psg, vf.peplx
    vf.glottal_area
    vf.contact_area
    vf.aspiration_noise.unoise
    lungs.plung
    vf.aspiration_noise.re2b


if __name__ == "__main__":
    test_sim()
