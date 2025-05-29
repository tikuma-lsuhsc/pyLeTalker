function [out] = test_lip(a,x)
    lip = LeTalkerLips(a);
    N = length(x);
    out = zeros(N,2);
    for n = 1:length(x)
        [out(n,1), out(n,2)] = lip.step(x(n));
    endfor
