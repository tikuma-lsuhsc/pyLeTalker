function y = test_vocaltract(areas, x, fs)
    model = LeTalkerVocalTract(areas, fs);
    N = length(x);
    y = zeros(N,3);
    for n = 1:length(x)
        [y(n,1),y(n,2),y(n,3)] = model.step(x(n));
    endfor
