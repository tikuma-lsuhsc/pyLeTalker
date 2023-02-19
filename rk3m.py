import numpy as np


def rk3m(y, dt, p):
    # ;
    # Fourth order Runge-Kutta algorithm for Three-mass model */
    # Brad H. Story    6-1-98            */

    dyo = eom_3m(y, p)

    k1 = dt * dyo

    dyo = eom_3m(y + k1, p)
    k2 = dt * dyo

    dyo = eom_3m(y + k2 / 2.0, p)
    k3 = dt * dyo

    dyo = eom_3m(y + k3, p)
    k4 = dt * dyo

    y += k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0

    return y


def eom_3m(yo, p):
    # Equations of motion for Three-Mass model
    # Author: Brad Story
    #
    # x1=yo[0] x2=yo[1] xb=yo[2]
    # x1dot=yo[3] x2dot=yo[4] xbdot=yo[5]
    # f1,f2: forces
    # b1,b2,B: damping constant (mass 1, mass 2, body mass)
    # k1,k2,kc,K: spring constant (mass1, mass2, shear-coupling, body mass

    dyo = np.array(
        [
            yo[3],  # velocity of mass 1
            yo[4],  # v of mass 2
            yo[5],  # v of body mass
            (
                p.f1
                - p.b1 * (yo[3] - yo[5])
                - p.k1 * ((yo[0]) - (yo[2]))
                - p.kc * ((yo[0]) - (yo[1]))
            )
            / p.m1,
            (
                p.f2
                - p.b2 * (yo[4] - yo[5])
                - p.k2 * ((yo[1]) - (yo[2]))
                - p.kc * ((yo[1]) - (yo[0]))
            )
            / p.m2,
            (
                p.k1 * ((yo[0]) - (yo[2]))
                + p.k2 * ((yo[1]) - (yo[2]))
                + p.b1 * (yo[3] - yo[5])
                + p.b2 * (yo[4] - yo[5])
                - p.K * (yo[2])
                - p.B * yo[5]
            )
            / p.M,
        ]
    )
    return dyo
