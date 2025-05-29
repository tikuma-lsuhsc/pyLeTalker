function [Fs, act, ata, alc, x01, x02, Lo, To, Dmo, Dlo, Dco, zeta, As, Ae] = LeTalker_init(N)

    Fs = 44100;
    InitializeLeTalker

    rand ("seed", 0);

    %LeTalker version 1.23
    %LeTalker - means "Lumped element Talker" and is the control code
    %for running the three-mass model of Story and Titze (1995) and Titze and Story (2002) with sub and supra glottal
    % airways;
    % Author: Brad Story, Univ. of Arizona
    %Date: 06.18.2014

    p.ar = p.ar(:);

    p = rules_consconv(p, c);

    %load filter coefficients for band pass filtering glottal noise
    load filter_coeffs;

    %If SUB=1 the subglottal system is included; SUB=0 will take it out.
    SUB = p.SUB;

    if (SUB == 0)
        p.ar(end) = 1000000;
    endif

    %If SUPRA=1 the supraglottal system is included; SUPRA=0 will take it out.
    SUPRA = p.SUPRA;

    if (SUPRA == 0)
        p.ar(1) = 1000000;
    endif

    neq = 6;
    F = zeros(neq, 1);
    r.theta = zeros(N, 1);
    r.x = zeros(N, 1);
    r.xb = zeros(N, 1);

    p.tan_th0 = p.xc / p.T;
    p.th0 = atan(p.tan_th0);

    for i = 1:6
        F(i) = 0;
    endfor

    %For use with Titze 1 below-----
    F(1) = 0;
    F(2) = 0;
    F(3) = 0;
    %For use with Titze 2 below-----
    % F(1) = p.x01;
    % F(2) = p.x02;
    % F(3) = p.xb0;

    %-----
    a = p.ar(:);
    Nsect = length(a);
    ieplx = 1;
    jmoth = 44;
    itrch = jmoth + 1;
    jtrch = Nsect;
    f = zeros(Nsect, 1);
    b = zeros(Nsect, 1);
    r1 = zeros(Nsect, 1);
    r2 = zeros(Nsect, 1);
    aprev = zeros(jmoth, 1);
    atot = zeros(jmoth, 1);

    r.x1 = []; %lower mass disp
    r.x2 = []; %upper mass disp
    r.xb = []; %body mass disp
    r.po = []; %output pressure
    r.ug = []; %glottal flow
    r.ga = []; %glottal area

    %---Initialize vectors for noise generation--
    xnois = zeros(5, 1);
    ynois = zeros(5, 1);
    Unois = 0;

    %vocal tract attenuation - same everywhere
    %alpha = .98*ones(Nsect);
    csnd = 35000;
    rho = 0.00114;
    rhoc = rho * csnd;
    mu = 0.000186;
    PI = pi;
    PI2 = 2 * PI;

    bprev = 0;
    fprev = 0;
    Pox = 0.0;

    dt = 1 / Fs;
    tme = 0.0;
    R = 128.0 / (9.0 * pi.^2);

    %--initialize----------
    p.a1 = 2 * p.L * F(1);
    p.a2 = 2 * p.L * F(2);
    p.ga = max(0, min(p.a1, p.a2));

    %vocal tract attenuation - based on xsect area
    alpha = 1 - p.vtatten ./ sqrt(a);

    %posterior glottal gap
    pgap = 0;
    p.psg = p.ps;

    %------

    tcos = 0.002;

    act = p.act
    ata = p.ata
    alc = p.alc
    x01 = p.x01
    x02 = p.x02
    Lo = c.Lo
    To = c.To
    Dmo = c.Dmo
    Dlo = c.Dlo
    Dco = c.Dco
    zeta = [0.1, 0.1, 0.6]
    As = p.As
    Ae = p.Ae
