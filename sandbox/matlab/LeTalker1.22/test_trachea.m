function y = test_trachea(areas, x)
    model = LeTalkerTrachea(areas);
    N = length(x);
    y = zeros(N, 1);

    bsg = 0.0
    for n = 1:N
        [y(n), z] = model.step(x(n), bsg);
    endfor
