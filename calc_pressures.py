def calc_pressures(p):
    #
    # Driving pressure calculations for the three-mass model based on Titze
    # 2002
    # Author: Brad Story
    # 10.25.11

    p.ke = (2 * p.ad / p.Ae) * (1 - p.ad / p.Ae)
    p.pkd = (p.psg - p.pe) / (1 - p.ke)

    # p.a1
    # p.a2
    p.ph = (p.psg + p.pe) / 2

    # --------------------------------------------------

    if p.a1 > p.delta and p.a2 <= p.delta:
        if p.zc >= p.zn:
            p.f1 = p.L * p.zn * p.psg
            p.f2 = p.L * (p.zc - p.zn) * p.psg + p.L * (p.T - p.zc) * p.ph
        else:
            p.f1 = p.L * p.zc * p.psg + p.L * (p.zn - p.zc) * p.ph
            p.f2 = p.L * (p.T - p.zn) * p.ph

    # --------------------------------------------------

    if p.a1 <= p.delta and p.a2 > p.delta:
        if p.zc < p.zn:
            p.f1 = p.L * p.zc * p.ph + p.L * (p.zn - p.zc) * p.pe
            p.f2 = p.L * (p.T - p.zn) * p.pe

        if p.zc >= p.zn:
            p.f1 = p.L * p.zn * p.ph
            p.f2 = p.L * (p.zc - p.zn) * p.ph + p.L * (p.T - p.zc) * p.pe

    # --------------------------------------------------

    if p.a1 <= p.delta and p.a2 <= p.delta:
        p.f1 = p.L * p.zn * p.ph
        p.f2 = p.L * (p.T - p.zn) * p.ph

    # -------------No contact -----------------------------------

    if p.a1 > p.delta and p.a2 > p.delta:
        if p.a1 < p.a2:
            if p.zd <= p.zn:
                p.f1 = (
                    p.L * p.zn * p.psg
                    - p.L * (p.zn - p.zd + (p.ad / p.a1) * p.zd) * p.pkd
                )
                p.f2 = p.L * (p.T - p.zn) * (p.psg - p.pkd)
            else:
                p.f1 = p.L * p.zn * (p.psg - (p.ad ** 2 / (p.an * p.a1)) * p.pkd)
                p.f2 = (
                    p.L * (p.T - p.zn) * p.psg
                    - p.L * ((p.T - p.zd) + (p.ad / p.an) * (p.zd - p.zn)) * p.pkd
                )

        elif p.a1 >= p.a2:
            p.f1 = p.L * p.zn * (p.psg - (p.a2 ** 2 / (p.an * p.a1)) * p.pkd)
            p.f2 = p.L * (p.T - p.zn) * (p.psg - (p.a2 / p.an) * p.pkd)

    return p
