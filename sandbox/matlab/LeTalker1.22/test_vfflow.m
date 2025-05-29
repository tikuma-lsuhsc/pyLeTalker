function [out, ug] = test_vfflow(ftrch, beplx, ga, Ae,At)
    vf = LeTalkerVFFlow(Ae,At);
    N = length(ga);
    out = zeros(N,2);
    ug = zeros(N,1);
    for n = 1:N
        [out(n,1), out(n,2), ug(n)] = vf.step(ftrch(n), beplx(n), ga(n));
    endfor
