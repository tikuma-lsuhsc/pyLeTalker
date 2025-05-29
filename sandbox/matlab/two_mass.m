%% Program for two-mass model proposed by Ishizaka and Flanagan (1972)
% with additional subglottal tract upstream of two-mass model.
% Separation point is predicted based on Pelorson et al.(1994).
%
% % MIT License
% %
% % Copyright (c) 2022 Tsukasa Yoshinaga
% %
% % Permission is hereby granted, free of charge, to any person obtaining a copy
% % of this software and associated documentation files (the "Software"), to deal
% % in the Software without restriction, including without limitation the rights
% % to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% % copies of the Software, and to permit persons to whom the Software is
% % furnished to do so, subject to the following conditions:
% %
% % The above copyright notice and this permission notice shall be included in all
% % copies or substantial portions of the Software.
% %
% % THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% % IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% % FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% % AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% % LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% % OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% % SOFTWARE.

% Initialization
clear;

pkg load statistics

% material properties
m1 = 0.17/1000; % g -> kg
m2 = 0.03/1000; % g -> kg
k1 = 50; % N/m
k2 = 8; %  N/m
k1c = 320; % N/m for collision
k2c = 32; % N/m for collision
kc = 25; % N/m coupring stiffness

% tension imbalance parameters
Qr = 1.0;
Ql = 1.0;

% damping coefficients
zeta1 = 0.1; zeta2 = 0.6;

% vocal fold dimensions
d1 = 2.5/1000; % mm -> m
d2 = 0.5/1000; % mm -> m
lg = 17/1000; % mm -> m
Ag01 = 11.5/1000000; %glottal area at rest (mm^2) 0,  11,  11.5
Ag02 = 12/1000000; %glottal area at rest (mm^2) 2,  11,   12
x1_min = Ag01 / (2 * lg);
x2_min = Ag02 / (2 * lg);

% parameters for tangential line calculation
xe = 0; % x at glottal outlet
xe2 = -d2 / 2; % position of mass2
xe1 =- (d2 + d1 / 2); %position of mass1
xe0 = xe1 * 3.5; %edge of tangential line (x = -6.125 mm)
yb = -5/1000; %m -> mm bottom surface
yt = 5/1000; %m -> mm  top surface
xc = linspace(xe0, xe, 62); %grid points
xsep = linspace(xe, -xe0, 62);
dx = xc(2) - xc(1); %dx = 0.1 mm
y1 = zeros(62, 1); theta1 = zeros(61, 1);
y2 = zeros(62, 1); theta2 = zeros(61, 1);
dhdx = zeros(1, 61); dddx = zeros(1, 61);
md1 = zeros(1, 61); md2 = zeros(1, 61); mh = zeros(1, 61);
AA = zeros(63, 1);
AA(1) = lg * (yt - yb); %area of glottal inlet

% Flow properties
Pl = 2000; % lung pressure (Pa)
rho = 1.205; % air density (kg/m3)
mu = 1.512 * 10 ^ -5; % viscosity of air
c = 343; % speed of sound (m/s)

% vocal tract area (0.5, 5 cm2 -> m2)
A1 = 0.5/100/100; A2 = 0.5/100/100; A3 = 4/100/100; A4 = 5/100/100;
% vocal tract length (4 + 4 + 4 + 5.5 cm -> m)
l1 = 4/100; l2 = 4/100; l3 = 4/100; l4 = 5.5/100;
% inlet chamber length (10 cm) and area (9.6 x 1.7 cm2 -> m2)
lsg1 = 10/100; Asg1 = 10 * 3.2/100/100;
% subglottal length (10 + 13.6 cm -> m)
lsg2 = 13.6/3/100; lsg3 = 13.6/3/100; lsg4 = 13.6/3/100;
% subglottal area (3 cm2 -> m2)
Asg2 = 3/100/100; Asg3 = 3/100/100; Asg4 = 3/100/100;

% Inductance, capacitance and resistance for subglottal tract
Lu1 = rho * lsg1 / (2 * Asg1); Lu2 = rho * lsg2 / (2 * Asg2);
Lu3 = rho * lsg3 / (2 * Asg3); Lu4 = rho * lsg2 / (2 * Asg4);
Cu1 = lsg1 * Asg1 / (rho * c ^ 2); Cu2 = lsg2 * Asg2 / (rho * c ^ 2);
Cu3 = lsg3 * Asg3 / (rho * c ^ 2); Cu4 = lsg2 * Asg4 / (rho * c ^ 2);
R2 = 0.14 / A1 ^ 2 * sqrt(rho * mu * sqrt(k1 / m1) / 2); % original 0.1

% Inductance, capacitance and resistance for supraglottal tract
L1 = rho * l1 / (2 * A1); L2 = rho * l2 / (2 * A2);
L3 = rho * l3 / (2 * A3); L4 = rho * l4 / (2 * A4);
C1 = l1 * A1 / (rho * c ^ 2); C2 = l2 * A2 / (rho * c ^ 2);
C3 = l3 * A3 / (rho * c ^ 2); C4 = l4 * A4 / (rho * c ^ 2);
Lr = 8 * rho / (3 * pi) * sqrt(pi * A4); Rr = 128 * rho * c / (9 * pi ^ 2 * A4);

% Time Step
delta_t = 0.25E-5;
total_sim_time = 0.05; %total secs
iterations = round(total_sim_time / delta_t); %Total no. of iterations
t = (1:iterations) * delta_t;

% Initial consitions
x11 = zeros(iterations, 1); x12 = zeros(iterations, 1);
x21 = zeros(iterations, 1); x22 = zeros(iterations, 1);
v11 = zeros(iterations, 1); v12 = zeros(iterations, 1);
v21 = zeros(iterations, 1); v22 = zeros(iterations, 1);
%
Ug = zeros(iterations, 1); U1 = zeros(iterations, 1);
U2 = zeros(iterations, 1); U3 = zeros(iterations, 1);
Ul = zeros(iterations, 1); Ur = zeros(iterations, 1);
Uu1 = zeros(iterations, 1); Uu2 = zeros(iterations, 1);
Uu3 = zeros(iterations, 1); Uu4 = zeros(iterations, 1);
%
Pg = zeros(iterations, 1);
Pp1 = zeros(iterations, 1); Pp2 = zeros(iterations, 1);
Ag1 = zeros(iterations, 1); Ag2 = zeros(iterations, 1);
Ag1(1) = Ag01; Ag2(1) = Ag02;

% start time loop
for i = 2:iterations
    %% tangentile line calculation for two-mass
    % right mass
    y11 = x11(i - 1) -x1_min - d1 / 2;
    y12 = x12(i - 1) -x2_min - d2 / 2;
    cr = (xe0 - xe1) ^ 2 + (yb - y11) ^ 2;
    xc1 = d1 / 2 * ((xe0 - xe1) * d1 / 2 + (yb - y11) * sqrt((xe0 - xe1) ^ 2 + (yb - y11) ^ 2 - d1 ^ 2/4)) / cr + xe1;
    yc1 = y11 + sqrt(d1 ^ 2/4 - (xc1 - xe1) ^ 2);

    cr = (xe2 - xe1) ^ 2 + (y12 - y11) ^ 2;
    xc2 = ((xe2 - xe1) * (d1 / 2 - d2 / 2) * d1 / 2 - (y12 - y11) * d1 / 2 * sqrt((xe2 - xe1) ^ 2 + (y12 - y11) ^ 2 - (d1 / 2 - d2 / 2) ^ 2)) / cr + xe1;
    yc2 = y11 + sqrt(d1 ^ 2/4 - (xc2 - xe1) ^ 2);
    xc3 = ((xe2 - xe1) * (d1 / 2 - d2 / 2) * d2 / 2 - (y12 - y11) * d2 / 2 * sqrt((xe2 - xe1) ^ 2 + (y12 - y11) ^ 2 - (d1 / 2 - d2 / 2) ^ 2)) / cr + xe2;
    yc3 = y12 + sqrt(d2 ^ 2/4 - (xc3 - xe2) ^ 2);

    for j = 1:length(xc)

        if xc(j) < xc1
            y1(j) = yb + (yc1 - yb) * (xc(j) - xe0) / (xc1 - xe0);
        elseif xc(j) < xc2
            y1(j) = y11 + sqrt(d1 ^ 2/4 - (xc(j) - xe1) ^ 2);
        elseif xc(j) < xc3
            y1(j) = yc2 + (yc3 - yc2) * (xc(j) - xc2) / (xc3 - xc2);
        elseif xc(j) <= xe
            y1(j) = y12 + sqrt(d2 ^ 2/4 - (xc(j) - xe2) ^ 2);
        end

    end

    % left mass
    y21 = x21(i - 1) +x1_min + d1 / 2;
    y22 = x22(i - 1) +x2_min + d2 / 2;
    cr = (xe0 - xe1) ^ 2 + (yt - y21) ^ 2;
    xc1 = d1 / 2 * ((xe0 - xe1) * d1 / 2 - (yt - y21) * sqrt((xe0 - xe1) ^ 2 + (yt - y21) ^ 2 - d1 ^ 2/4)) / cr + xe1;
    yc1 = y21 - sqrt(d1 ^ 2/4 - (xc1 - xe1) ^ 2);

    cr = (xe2 - xe1) ^ 2 + (y22 - y21) ^ 2;
    xc2 = ((xe2 - xe1) * (d1 / 2 - d2 / 2) * d1 / 2 + (y22 - y21) * d1 / 2 * sqrt((xe2 - xe1) ^ 2 + (y22 - y21) ^ 2 - (d1 / 2 - d2 / 2) ^ 2)) / cr + xe1;
    yc2 = y21 - sqrt(d1 ^ 2/4 - (xc2 - xe1) ^ 2);
    xc3 = ((xe2 - xe1) * (d1 / 2 - d2 / 2) * d2 / 2 + (y22 - y21) * d2 / 2 * sqrt((xe2 - xe1) ^ 2 + (y22 - y21) ^ 2 - (d1 / 2 - d2 / 2) ^ 2)) / cr + xe2;
    yc3 = y22 - sqrt(d2 ^ 2/4 - (xc3 - xe2) ^ 2);

    for j = 1:length(xc)

        if xc(j) < xc1
            y2(j) = yt + (yc1 - yt) * (xc(j) - xe0) / (xc1 - xe0);
        elseif xc(j) < xc2
            y2(j) = y21 - sqrt(d1 ^ 2/4 - (xc(j) - xe1) ^ 2);
        elseif xc(j) < xc3
            y2(j) = yc2 + (yc3 - yc2) * (xc(j) - xc2) / (xc3 - xc2);
        elseif xc(j) <= xe
            y2(j) = y22 - sqrt(d2 ^ 2/4 - (xc(j) - xe2) ^ 2);
        end

    end

    % area for tangential lines
    AAtmp = (y2 - y1) * lg;
    negative = find(AAtmp < 0);
    AAtmp(negative) = zeros(size(negative));
    AA(2:end) = AAtmp;

    % angle at each point
    for j = 1:length(xc) - 1
        theta1(j) = atan((y1(j + 1) - y1(j)) / (xc(j + 1) - xc(j)));
        theta2(j) = atan(- (y2(j + 1) - y2(j)) / (xc(j + 1) - xc(j)));
    end

    %% calculation of glottal flowrate Ug
    %subglottal tract
    Uu1(i) = Uu1(i - 1) - delta_t / (Lu1) * (delta_t / Cu1 * sum(Uu1(1:i - 1) - Uu2(1:i - 1)) - Pl);
    Uu2(i) = Uu2(i - 1) - delta_t / (Lu1 + Lu2) * (delta_t / Cu2 * sum(Uu2(1:i - 1) - Uu3(1:i - 1)) ...
        +delta_t / Cu1 * sum(Uu2(1:i - 1) - Uu1(1:i - 1)) + R2 * Uu2(i - 1));
    Uu3(i) = Uu3(i - 1) - delta_t / (Lu2 + Lu3) * (delta_t / Cu3 * sum(Uu3(1:i - 1) - Uu4(1:i - 1)) ...
        +delta_t / Cu2 * sum(Uu3(1:i - 1) - Uu2(1:i - 1)) + R2 * Uu3(i - 1));
    Uu4(i) = Uu4(i - 1) - delta_t / (Lu3 + Lu4) * (delta_t / Cu4 * sum(Uu4(1:i - 1) - Ug(1:i - 1)) ...
        +delta_t / Cu3 * sum(Uu4(1:i - 1) - Uu3(1:i - 1)) + R2 * Uu4(i - 1));

    %glottal flow rate Ug
    Lg1 = rho * d1 / Ag1(i - 1); Lg2 = rho * d2 / Ag2(i - 1);
    Rk1 = 0.19 * rho / Ag1(i - 1) ^ 2; Rk2 = rho * 0.5 / Ag2(i - 1) ^ 2;
    Rv1 = 12 * mu * lg ^ 2 * d1 / (Ag1(i - 1) ^ 3); Rv2 = 12 * mu * lg ^ 2 * d2 / (Ag2(i - 1) ^ 3);

    if Ag1(i - 1) > 0 && Ag2(i - 1) > 0
        % newton raphson method
        for j = 1:100
            F = (Rk1 + Rk2) * abs(Ug(i)) * Ug(i) + (Rv1 + Rv2) * Ug(i) + (Lg1 + Lg2 + L1 + Lu4) * (Ug(i) - Ug(i - 1)) / delta_t ...
                +delta_t / C1 * sum(Ug(1:i - 1) - U1(1:i - 1)) + delta_t / Cu4 * sum(Ug(1:i) - Uu4(1:i));

            if (abs(F) < 1E-9)
                break
            end

            Fd = 2 * (Rk1 + Rk2) * Ug(i) + Rv1 + Rv2 + (Lg1 + Lg2 + L1 + Lu4) / delta_t;
            Ug(i) = Ug(i) - F / Fd;
        end

        if j >= 100
            disp("error")
            return
        end

    else
        Ug(i) = 0;
    end

    %subglottal pressure
    Pg(i) = delta_t / Cu4 * sum(Uu4(1:i) - Ug(1:i));

    %supraglottal tract
    U1(i) = U1(i - 1) - delta_t / (L1 + L2) * (delta_t / C2 * sum(U1(1:i - 1) - U2(1:i - 1)) ...
        +delta_t / C1 * sum(U1(1:i - 1) - Ug(1:i - 1)));
    U2(i) = U2(i - 1) - delta_t / (L2 + L3) * (delta_t / C3 * sum(U2(1:i - 1) - U3(1:i - 1)) ...
        +delta_t / C2 * sum(U2(1:i - 1) - U1(1:i - 1)));
    U3(i) = U3(i - 1) - delta_t / (L3 + L4) * (delta_t / C4 * sum(U3(1:i - 1) - Ul(1:i - 1)) ...
        +delta_t / C3 * sum(U3(1:i - 1) - U2(1:i - 1)));
    Ul(i) = ((L4 + Lr) / delta_t * Ul(i - 1) - Lr / delta_t * Ur(i - 1) - delta_t / C4 * sum(Ul(1:i - 1) - U3(1:i - 1)) ...
        + (Lr / delta_t) ^ 2 / (Lr / delta_t + Rr) * (Ur(i - 1) - Ul(i - 1))) ...
        / ((L4 + Lr) / delta_t - (Lr / delta_t) ^ 2 / (Lr / delta_t + Rr));
    Ur(i) = Lr / delta_t * (Ul(i) + Ur(i - 1) - Ul(i - 1)) / (Lr / delta_t + Rr);

    %% pressure calculation
    % initial values for force and pressure
    FF1 = 0; FF2 = 0;
    Pt = zeros(63, 1); Pt(1) = Pg(i);

    % separation prediction based on Pelorson et al. (1994)
    h = AA(2:end) / lg;
    Re = Ug(i) / (mu / rho * lg);
    delta2 = 0.5 * (-5 * 0.6641 ^ 2 / Re * xsep + sqrt((5 * 0.6641 ^ 2 / Re) ^ 2 * xsep .* xsep ...
        +4 * (0.6641 ^ 2 / Re * h.' .* xsep + 0.015 / Re * (d2 / 2 / min(h)) ^ 0.5)));
    delta1 = 2.5 * delta2;

    for k = 1:length(h) - 1
        dhdx(k) = (h(k + 1) - h(k)) / dx;
        dddx(k) = (delta1(k + 1) - delta1(k)) / dx;
        md1(k) = (delta1(k + 1) + delta1(k)) / 2;
        md2(k) = (delta2(k + 1) + delta2(k)) / 2;
        mh(k) = (h(k + 1) + h(k)) / 2;
    end

    ramda = -md2 .^ 2 * Re ./ (mh - 2 * md1) .^ 2 .* (dhdx - 2 * dddx);
    sep = knnsearch(real(ramda).', -0.0992);

    if sep == 1
        sep = length(xc) + 1;
    end

    % surface pressure distribution Pt
    for j = 1:length(xc)
        heq = (AA(j) + AA(j + 1)) / (2 * lg);

        if AA(j) > 0 && AA(j + 1) > 0 && j <= sep
            Pt(j + 1) = Pt(j) + 0.5 * rho * Ug(i) ^ 2 * (1 / AA(j) ^ 2 - 1 / (AA(j + 1)) ^ 2) - 12 * mu * dx / (lg * (heq) ^ 3) * Ug(i);
        elseif AA(j) > 0 && AA(j + 1) > 0 && j > sep
            Pt(j + 1) = Pt(j);
        elseif AA(j) > 0 && AA(j + 1) <= 0
            Pt(j + 1) = 0;
            break
        end

        if xc(j) < xe - d2 && xc(j) > xe0
            FF1 = FF1 + cos(theta2(j - 1)) * Pt(j + 1) * dx * lg;
        elseif xc(j) >= xe - d2
            FF2 = FF2 + cos(theta2(j - 1)) * Pt(j + 1) * dx * lg;
        end

    end

    % pressures for mass 1 and 2
    Pp1(i) = FF1 / (lg * (-xe0 - d2));
    Pp2(i) = FF2 / (lg * d2);

    %% constant preparation for equation of motion
    if x21(i - 1) - x11(i - 1) > -x1_min * 2 && x22(i - 1) - x12(i - 1) > -x2_min * 2
        s1 = k1; s2 = k2;
        r1 = 2 * zeta1 * sqrt(m1 * k1);
        r2 = 2 * zeta2 * sqrt(m2 * k2);
    elseif x21(i - 1) - x11(i - 1) <= -x1_min * 2 && x22(i - 1) - x12(i - 1) > -x2_min * 2
        s1 = k1c; s2 = k2;
        r1 = 2 * (zeta1 + 1) * sqrt(m1 * k1);
        r2 = 2 * zeta2 * sqrt(m2 * k2);
    elseif x21(i - 1) - x11(i - 1) > -x1_min * 2 && x22(i - 1) - x12(i - 1) <= -x2_min * 2
        s1 = k1; s2 = k2c;
        r1 = 2 * zeta1 * sqrt(m1 * k1);
        r2 = 2 * (zeta2 + 1) * sqrt(m2 * k2);
    else
        s1 = k1c; s2 = k2c;
        r1 = 2 * (zeta1 + 1) * sqrt(m1 * k1);
        r2 = 2 * (zeta2 + 1) * sqrt(m2 * k2);
    end

    %% equation of motion for right mass (Runge-Kutta method)
    L11 = delta_t * v11(i - 1);
    K11 = (delta_t / m1 * Qr) * (-FF1 - r1 * v11(i - 1) - s1 * Qr * x11(i - 1) - kc * Qr * (x11(i - 1) - x12(i - 1)));
    L12 = delta_t * v12(i - 1);
    K12 = (delta_t / m2 * Qr) * (-FF2 - r2 * v12(i - 1) - s2 * Qr * x12(i - 1) - kc * Qr * (x12(i - 1) - x11(i - 1)));
    L21 = delta_t * (v11(i - 1) + K11 / 2);
    K21 = (delta_t / m1 * Qr) * (-FF1 - r1 * (v11(i - 1) + K11 / 2) - s1 * Qr * (x11(i - 1) + L11 / 2) - kc * Qr * ((x11(i - 1) + L11 / 2) - (x12(i - 1) + L12 / 2)));
    L22 = delta_t * (v12(i - 1) + K12 / 2);
    K22 = (delta_t / m2 * Qr) * (-FF2 - r2 * (v12(i - 1) + K12 / 2) - s2 * Qr * (x12(i - 1) + L12 / 2) - kc * Qr * ((x12(i - 1) + L12 / 2) - (x11(i - 1) + L11 / 2)));
    L31 = delta_t * (v11(i - 1) + K21 / 2);
    K31 = (delta_t / m1 * Qr) * (-FF1 - r1 * (v11(i - 1) + K21 / 2) - s1 * Qr * (x11(i - 1) + L21 / 2) - kc * Qr * ((x11(i - 1) + L21 / 2) - (x12(i - 1) + L22 / 2)));
    L32 = delta_t * (v12(i - 1) + K22 / 2);
    K32 = (delta_t / m2 * Qr) * (-FF2 - r2 * (v12(i - 1) + K22 / 2) - s2 * Qr * (x12(i - 1) + L22 / 2) - kc * Qr * ((x12(i - 1) + L22 / 2) - (x11(i - 1) + L21 / 2)));
    L41 = delta_t * (v11(i - 1) + K31);
    K41 = (delta_t / m1 * Qr) * (-FF1 - r1 * (v11(i - 1) + K31) - s1 * Qr * (x11(i - 1) + L31) - kc * Qr * ((x11(i - 1) + L31) - (x12(i - 1) + L32)));
    L42 = delta_t * (v12(i - 1) + K32);
    K42 = (delta_t / m2 * Qr) * (-FF2 - r2 * (v12(i - 1) + K32) - s2 * Qr * (x12(i - 1) + L32) - kc * Qr * ((x12(i - 1) + L32) - (x11(i - 1) + L31)));

    % velocity and displacement
    x11(i) = x11(i - 1) + (1/6) * (L11 + 2 * L21 + 2 * L31 + L41);
    v11(i) = v11(i - 1) + (1/6) * (K11 + 2 * K21 + 2 * K31 + K41);
    x12(i) = x12(i - 1) + (1/6) * (L12 + 2 * L22 + 2 * L32 + L42);
    v12(i) = v12(i - 1) + (1/6) * (K12 + 2 * K22 + 2 * K32 + K42);

    %% equation of motion for left mass
    L11 =                          delta_t * v21(i - 1);
    K11 =  (delta_t / m1 * Ql) * (FF1 - r1 * v21(i - 1)            - s1 * Ql * x21(i - 1)             - kc * Ql * (x21(i - 1) - x22(i - 1)));
    L12 =                          delta_t * v22(i - 1);
    K12 =  (delta_t / m2 * Ql) * (FF2 - r2 * v22(i - 1)            - s2 * Ql * x22(i - 1)             - kc * Ql * (x22(i - 1) - x21(i - 1)));
    L21 =                         delta_t * (v21(i - 1) + K11 / 2);
    K21 = (delta_t / m1 * Ql) * (FF1 - r1 * (v21(i - 1) + K11 / 2) - s1 * Ql * (x21(i - 1) + L11 / 2) - kc * Ql * ((x21(i - 1) + L11 / 2) - (x22(i - 1) + L12 / 2)));
    L22 =                         delta_t * (v22(i - 1) + K12 / 2);
    K22 = (delta_t / m2 * Ql) * (FF2 - r2 * (v22(i - 1) + K12 / 2) - s2 * Ql * (x22(i - 1) + L12 / 2) - kc * Ql * ((x22(i - 1) + L12 / 2) - (x21(i - 1) + L11 / 2)));
    L31 =                         delta_t * (v21(i - 1) + K21 / 2);
    K31 = (delta_t / m1 * Ql) * (FF1 - r1 * (v21(i - 1) + K21 / 2) - s1 * Ql * (x21(i - 1) + L21 / 2) - kc * Ql * ((x21(i - 1) + L21 / 2) - (x22(i - 1) + L22 / 2)));
    L32 =                         delta_t * (v22(i - 1) + K22 / 2);
    K32 = (delta_t / m2 * Ql) * (FF2 - r2 * (v22(i - 1) + K22 / 2) - s2 * Ql * (x22(i - 1) + L22 / 2) - kc * Ql * ((x22(i - 1) + L22 / 2) - (x21(i - 1) + L21 / 2)));
    L41 =                         delta_t * (v21(i - 1) + K31);
    K41 = (delta_t / m1 * Ql) * (FF1 - r1 * (v21(i - 1) + K31)     - s1 * Ql * (x21(i - 1) + L31)     - kc * Ql * ((x21(i - 1) + L31)     - (x22(i - 1) + L32)));
    L42 =                         delta_t * (v22(i - 1) + K32);
    K42 = (delta_t / m2 * Ql) * (FF2 - r2 * (v22(i - 1) + K32)     - s2 * Ql * (x22(i - 1) + L32)     - kc * Ql * ((x22(i - 1) + L32)     - (x21(i - 1) + L31)));

    % velocity and displacement
    x21(i) = x21(i - 1) + (1/6) * (L11 + 2 * L21 + 2 * L31 + L41);
    v21(i) = v21(i - 1) + (1/6) * (K11 + 2 * K21 + 2 * K31 + K41);
    x22(i) = x22(i - 1) + (1/6) * (L12 + 2 * L22 + 2 * L32 + L42);
    v22(i) = v22(i - 1) + (1/6) * (K12 + 2 * K22 + 2 * K32 + K42);

    % area for next step
    Ag1(i) = lg * ((x21(i) + x1_min) - (x11(i) - x1_min));
    Ag2(i) = lg * ((x22(i) + x2_min) - (x12(i) - x2_min));
    if Ag1(i) < 0; Ag1(i) = 0; end
    if Ag2(i) < 0; Ag2(i) = 0; end

end

% f = figure;
% f.Position(4) = 700;

% % displacement
% subplot(3, 1, 1)
% plot(t * 1000, (x21 + x1_min) * 1000, 'b')
% hold on
% plot(t * 1000, (x22 + x2_min) * 1000, 'r')
% plot(t * 1000, (x11 - x1_min) * 1000, 'b--')
% plot(t * 1000, (x12 - x2_min) * 1000, 'r--')
% x = [0 0; total_sim_time * 1000 0];
% plot(x(:, 1), x(:, 2), 'k:')
% lgd = legend('y_1_L', 'y_2_L', 'y_1_R', 'y_2_R');
% lgd.Orientation = 'horizontal';
% lgd.Location = 'northoutside';
% xlabel('t (ms)')
% ylabel('Displacement (mm)')

% % pressure on each mass
% subplot(3, 1, 2)
% plot(t * 1000, Pp1 / 1000, 'b')
% hold on
% plot(t * 1000, Pp2 / 1000, 'r')
% xlabel('t (ms)')
% ylabel('Pressure (kPa)')
% lgd2 = legend('Mass1', 'Mass2');
% lgd2.Orientation = 'horizontal';

% % flowrate
% subplot(3, 1, 3)
% plot(t * 1000, Ug * 1000000, 'b')
% xlabel('t (ms)')
% ylabel('Ug (cm^3/s)')

% %end program two_mass.m
