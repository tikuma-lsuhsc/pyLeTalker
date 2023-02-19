# this is a script that defines a structure p
# structure contains all variables for the 3-mass
# and bar-plate models
#
# Brad Story 10.25.11
# Takeshi Ikuma 02.18.23 - Python transplant

import numpy as np
from scipy.io.matlab import loadmat
from dataclasses import dataclass


# constants
@dataclass(frozen=True)
class Constants:
    mu = 5000  # shear modulus in cover
    mub = 5000
    muc = 5000  # shear modulus in body
    rho = 1.04  # tissue density
    R = 3.0  # torque ratio
    G = 0.2  # gain of elongation
    H = 0.2  # adductory strain factor
    Lo = 1.6  # cadaveric rest length for male
    # Lo = 1  # cadaveric rest length for female
    To = 0.3  # resting thickness -old value 0.45 cm
    Dmo = 0.4  # resting depth of muscle
    Dlo = 0.2  # resting depth of deep layer
    Dco = 0.2  # resting depth of the cover
    zeta = 0.1
    Csound = 35000.0  # speed of sound
    rhoc = 39.9  # rho * speed of sound for air


@dataclass
class Properties:
    ic = 1  # moment of inertia
    bc = 1  # rotational damping
    kr = 1  # rotational stiffness
    ta = 1  # aerodynamic torque
    fa = 1  # force at the nodal point
    M = 1  # body mass
    B = 1  # body damping
    K = 1  # body stiffness
    m = 1  # cover mass
    b = 1  # cover damping
    k = 1  # cover stiffness
    x02 = 0.0  # upper prephonatory displacement
    x01 = 0.0005
    th0 = 1  # prephonatory convergence angle
    x0 = 1
    xb0 = 0.6
    T = 1  # thickness of the vocal folds
    L = 1  # length of the vocal folds
    Dc = 0  # depth of cover
    Db = 0  # depth of body
    zn = 0  # nodal point as measured from the bottom of the vocal folds
    a1 = 0  # glottal entry area
    a2 = 0  # glottal exit area
    ad = 0
    an = 0  # area at the nodal point
    ga = 0  # glottal area
    ca = 0  # contact area
    m1 = 0  # lower mass
    m2 = 0  # upper mass
    k1 = 0  # lower stiffness
    k2 = 0  # upper stiffness
    kc = 0  # coupling stiffness
    b1 = 0  # lower damping coeff
    b2 = 0  # upper damping coeff
    pg = 0  # mean intraglottal pressure
    p1 = 0  # pressure on lower mass
    p2 = 0  # pressure on upper mass
    ps = 0  # subglottal pressure
    pkd = 0
    PL = 7840  # respirtory driving pressure from lungs (dyn/cm^2)
    ph = 0  # hydrostatic pressure
    pe = 0  # supraglottal (epilaryngeal) pressure
    psg = 0
    f1 = 0  # force on lower mass
    f2 = 0  # force on upper mass
    u = 0  # glottal flow
    zc = 0  # the contact point

    act = 0.25  # cricothyroid activity, can range from 0 to 1
    ata = 0.25  # thyroarytenoid activity, can range from 0 to 1
    alc = 0.5  # lca activity
    apc = 0  # pca activity
    eps = 0  # strain
    xc = 0  # glottal convergence
    xn = 0  # xnode
    zc = 0
    x1 = 0
    x2 = 0
    xn0 = 0  # initial position of xnode

    sigb = 1  # body fiber stress
    sigl = 1  # passive ligament stress
    sigm = 1  # muscle stress
    sigam = 1050000  # max active muscle stress
    sigp = 1  # passive muscle stress
    sigmuc = 1
    sigc = 1

    Al = 0.05  # cross sectional area of ligament
    Am = 0.03  # cross sectional area of muscle
    Ac = 0.0195  # cross sectional area of cover

    Fob = 1
    Foc1 = 1
    Foc2 = 1

    b = 0  # quadratic coeff for muscle stress
    epsm = 0  # optimum strain

    delta = 0.0000001
    Ae = 1000000.0
    As = 1000000

    # Initialize vocal tract and trachea. Let supraglottal be 5 cm2 uniform
    # tube per Story and Titze 1995
    ar = np.full(76, 3.0)
    # load bs_origvowels.mat
    # ar[:44] = aa #(for an "ah" vowel)
    # ar[:44] = ii #(for an "ee" vowel)
    # ar[:44] = uumod #(for an "oo" vowel)

    # load and assign trachea. 1.5 cm2 tube that tapers to 0.3 cm2 just below
    # glottis
    # load trach.mat trchareas
    # ar[44:76] = trchareas
    ar[44:76] = loadmat("LeTalkerTrachea")["trach"].reshape(-1)

    # vtatten = 0.0015
    vtatten = 0.005
    SUB = 1
    SUPRA = 1
    Noise = 0

    # noise parameters
