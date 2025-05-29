function dyo = eom_3m(yo, dyo, tt, p)
    %Equations of motion for Three-Mass model
    %Author: Brad Story
    %
    %x1=yo(1); x2=yo(2); xb=yo(3);
    %x1dot=yo(4); x2dot=yo(5); xbdot=yo(6)
    % f1,f2: forces
    % b1,b2,B: damping constant (mass 1, mass 2, body mass)
    % k1,k2,kc,K: spring constant (mass1, mass2, shear-coupling, body mass

    dyo(1) = yo(4); % velocity of mass 1
    dyo(2) = yo(5); % v of mass 2
    dyo(3) = yo(6); % v of body mass

    dyo(4) = (p.f1 - p.b1 * (yo(4) - yo(6)) - p.k1 * ((yo(1)) - (yo(3))) - p.kc * ((yo(1)) - (yo(2)))) / p.m1;
    dyo(5) = (p.f2 - p.b2 * (yo(5) - yo(6)) - p.k2 * ((yo(2)) - (yo(3))) - p.kc * ((yo(2)) - (yo(1)))) / p.m2;
    dyo(6) = (p.k1 * ((yo(1)) - (yo(3))) + p.k2 * ((yo(2)) - (yo(3))) + p.b1 * (yo(4) - yo(6)) + p.b2 * (yo(5) - yo(6)) - p.K * (yo(3)) - p.B * yo(6)) / p.M;
endfunction
