function [dyo] = test_eom(yo, dyo, tt, m, k, kc, b, f)

    p.f1 = f(1);
    p.b1 = b(1);
    p.k1 = k(1);
    p.kc = kc;
    p.m1 = m(1);
    p.f2 = f(2);
    p.b2 = b(2);
    p.k2 = k(2);
    p.m2 = m(2);
    p.K = k(3);
    p.B = b(3);
    p.M = m(3);

    dyo = eom_3m(yo, dyo, tt, p);

end
