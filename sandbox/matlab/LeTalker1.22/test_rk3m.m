function p = test_rk3m(eps)

    p.eps = eps;

    [y] = rk3m(6, y, time, dt, p)

    sigmuc = p.sigmuc
    sigl = p.sigl
    sigp = p.sigp

end
