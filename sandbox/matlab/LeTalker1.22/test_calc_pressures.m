function p = test_calc_pressures(psg, pe, xi1, xi2, Ae, zn, L, T, delta)

    znot = zn / T;
    xn = (1 - znot) * xi1 + znot * xi2;
    tangent = (xi1 - xi2) / T;
    % tangent = (x01 - x02 + 2 * (F(1) - F(2))) / T;
    x1 = xn - (-zn) * tangent;
    x2 = xn - (T - zn) * tangent;
    p.zc = min(T, max(0., zn + xn / tangent));
    p.a1 = max(delta, 2 * L * x1);
    p.a2 = max(delta, 2 * L * x2);
    p.an = max(delta, 2 * L * xn);
    p.zd = min(T, max(0, -0.2 * x1 / tangent));
    p.ad = min(p.a2, 1.2 * p.a1);

    p.L = L;
    p.T = T;
    p.delta = delta;
    p.Ae = Ae;
    p.zn = zn;

    p = calc_pressures(psg, pe, p);

endfunction
