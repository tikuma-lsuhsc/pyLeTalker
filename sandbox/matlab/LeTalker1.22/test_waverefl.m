function y = test_waverefl(areas, x)
    model = LeTalkerTract(areas);
    N = length(x);
    y = zeros(N,2);
    for n = 1:length(x)
        [y(n,1),y(n,2)] = model.step(x(n,1), x(n,2));
    endfor
