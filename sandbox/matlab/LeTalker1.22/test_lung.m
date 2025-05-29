function f = test_lung(PL, beta, b)
    lung = LeTalkerLung(PL, beta);
    N = length(b);
    f = zeros(N,1);
    for n = 1:length(b)
        f(n) = lung.step(n, b(n));
    endfor
