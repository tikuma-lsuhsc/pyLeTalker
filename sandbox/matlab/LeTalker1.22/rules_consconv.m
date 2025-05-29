function p = rules_consconv(p, c)
    % Rules based on Titze and Story 2002, but with constant convergence ("near
    % rectangular" prephonatory glottal configuration).
    % Author: Brad Story
    % Rules start on p. 1069
    %

    p.eps = c.G * (c.R * p.act - p.ata) - c.H * p.alc; % strain (55)
    p.L = c.Lo * (1 + p.eps); % vibrating length
    p.T = c.To / (1 + 0.8 * p.eps); % thickness (59)

    Xcoef = 1;

    %----as written in the paper-----
    p.Db = (p.ata * c.Dmo + 0.5 * c.Dlo) / (1 + 0.2 * p.eps); % Depth (60)
    p.Dc = (c.Dco + 0.5 * c.Dlo) / (1 + 0.2 * p.eps); % Depth (61)

    p.zn = p.T * (1 + p.ata) / 3; % Nodal Point Rule (58)

    %-------------------
    %p.x01 = p.xc + p.x02;

    p.tan_th0 = p.xc / p.T;
    p.xn0 = p.x02 + (p.T - p.zn) * p.tan_th0;
    p.th0 = atan(p.tan_th0);

    %--------------------just for testing-these override rules for x02 and x01
    % p.x02 = 0.00;                          % initial displacement of top mass
    % p.xc = .0005;                          % convergence
    % p.x01 = 0.0005;                        % initial displacement of bottom mass
    %p.T = 0.3;
    %p.zn = 0.15;
    %-------------------------

    p = calc_stress(p);
    p.sigm = (p.ata) * p.sigam * max(0, 1 - 1.07 * (p.eps - 0.4)^2) + p.sigp;

    %----------As in the paper-------------------------------
    p.sigb = (0.5 * p.sigl * c.Dlo + p.sigm * c.Dmo) / p.Db;
    p.sigc = (p.sigmuc * c.Dco + 0.5 * p.sigl * c.Dlo) / p.Dc;

    EffL = p.L; % effective length

    % bar-plate model, not used for three-mass model
    p.ic = (c.rho * p.L * p.Dc * (p.T)^3) * (1/3 - (p.zn / p.T) * (1 - p.zn / p.T));
    p.m = c.rho * EffL * p.T * p.Dc;
    p.k = 2 * c.muc * EffL * p.T / p.Dc + pi^2 * p.sigc * (p.Dc / EffL) * p.T;
    p.kr = 0.5 * c.muc * EffL * p.T * p.Dc + (pi^2 * p.ic * p.sigc) / (EffL^2 * c.rho);

    p.M = c.rho * EffL * p.T * p.Db; % mass of body
    p.K = 2 * c.mub * EffL * p.T / p.Db + pi^2 * p.sigb * p.Db * p.T / EffL; % stiffness of body

    p.k1 = 2 * c.muc * (EffL * p.T / p.Dc) * p.zn / p.T + pi^2 * p.sigc * (p.Dc / EffL) * p.zn; % (12)
    p.k2 = 2 * c.muc * (EffL * p.T / p.Dc) * (1 - p.zn / p.T) + pi^2 * p.sigc * (p.Dc / EffL) * p.T * (1 - p.zn / p.T); % (13)
    p.m1 = c.rho * EffL * p.T * p.Dc * p.zn / p.T; % using (30) for m
    p.m2 = c.rho * EffL * p.T * p.Dc * (1 - p.zn / p.T);

    %Equation 13
    a = p.zn / p.T;
    p.kc = (0.5 * c.muc * (EffL * p.Dc / p.T) / (1/3 - a * (1 - a)) - 2 * c.muc * (EffL * p.T / p.Dc)) * a * (1 - a);

    p.bc = 2 * c.zeta * sqrt(p.ic * p.kr); %Check to see if the same zeta is used for rot. damping
    p.b = 2 * c.zeta * sqrt(p.m * p.k);
    p.B = 2 * (0.1) * sqrt(p.M * p.K);
    p.b1 = 2 * (0.1) * sqrt(p.m1 * p.k1);
    p.b2 = 2 * (0.6) * sqrt(p.m2 * p.k2);
