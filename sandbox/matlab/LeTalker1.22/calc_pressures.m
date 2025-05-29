function p = calc_pressures(psg, pe, p)
    %
    % Driving pressure calculations for the three-mass model based on Titze
    % 2002
    % Author: Brad Story
    %10.25.11

    p.ke = (2 * p.ad / p.Ae) * (1 - p.ad / p.Ae);
    p.pkd = (psg - pe) / (1 - p.ke);

    %p.a1
    %p.a2
    p.ph = (psg + pe) / 2;

    %--------------------------------------------------

    if (p.a1 > p.delta && p.a2 <= p.delta)

        if (p.zc >= p.zn)
            p.f1 = p.L * p.zn * psg;
            p.f2 = p.L * (p.zc - p.zn) * psg + p.L * (p.T - p.zc) * p.ph;
        else
            p.f1 = p.L * p.zc * psg + p.L * (p.zn - p.zc) * p.ph;
            p.f2 = p.L * (p.T - p.zn) * p.ph;
        end

    end

    %--------------------------------------------------

    if (p.a1 <= p.delta && p.a2 > p.delta)

        if (p.zc < p.zn)
            p.f1 = p.L * p.zc * p.ph + p.L * (p.zn - p.zc) * pe;
            p.f2 = p.L * (p.T - p.zn) * pe;
        end

        if (p.zc >= p.zn)
            p.f1 = p.L * p.zn * p.ph;
            p.f2 = p.L * (p.zc - p.zn) * p.ph + p.L * (p.T - p.zc) * pe;
        end

    end

    %--------------------------------------------------

    if (p.a1 <= p.delta && p.a2 <= p.delta)

        p.f1 = p.L * p.zn * p.ph;
        p.f2 = p.L * (p.T - p.zn) * p.ph;

    end

    %-------------No contact -----------------------------------

    if (p.a1 > p.delta && p.a2 > p.delta)

        if (p.a1 < p.a2)

            if (p.zd <= p.zn)
                p.f1 = p.L * p.zn * psg - p.L * (p.zn - p.zd + (p.ad / p.a1) * p.zd) * p.pkd;
                p.f2 = p.L * (p.T - p.zn) * (psg - p.pkd);
            else
                p.f1 = p.L * p.zn * (psg - (p.ad ^ 2 / (p.an * p.a1)) * p.pkd);
                p.f2 = p.L * (p.T - p.zn) * psg - p.L * ((p.T - p.zd) + (p.ad / p.an) * (p.zd - p.zn)) * p.pkd;
            end

        elseif (p.a1 >= p.a2)

            p.f1 = p.L * p.zn * (psg - (p.a2 ^ 2 / (p.an * p.a1)) * p.pkd);
            p.f2 = p.L * (p.T - p.zn) * (psg - (p.a2 / p.an) * p.pkd);
        end

    end
