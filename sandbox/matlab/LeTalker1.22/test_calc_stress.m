function p = test_calc_stress(eps)

    p.eps = eps;

    p = calc_stress(p);

    sigmuc = p.sigmuc
    sigl = p.sigl
    sigp = p.sigp

end
