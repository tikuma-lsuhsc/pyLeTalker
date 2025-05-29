classdef LeTalkerLips < handle
    %------------- Lip Radiation -----------  */
    % The radiation load approximation  of a piston in a infinite baffle proposed by Flanagan (1972)

    properties (Access='protected')
        s% states: previous f, previous b, previous Pout
        A% state-space 3x3 A matrix
        b% state-space 3x1 b matrix (input: current f)

        a1
        a2
        b1
        b2
    endproperties

    methods

        function obj = LeTalkerLips(a, fs, R, csnd)
            % a - between-lip cross-sectional area
            % atten - vocal tract attenuation factor (default: 5e-3)

            if nargin == 0
                return
            endif

            if nargin < 2
                fs = 44100;
            endif

            if nargin < 3 || R <= 0
                R = 128.0 / (9.0 * pi^2);
            endif

            if nargin < 4
                csnd = 35000;
            endif

            am = sqrt(a / pi);
            L = (2.0 * fs) * 8.0 * am / (3.0 * pi * csnd);
            a2 = -R - L + R * L;
            a1 = -R + L - R * L;
            b2 = R + L + R * L;
            b1 = -R + L + R * L;

            % b(jmoth) = (1/b2)*(f(jmoth)*a2 + fprev*a1 + bprev*b1);
            % Pout = (1/b2)*(Pox*b1 + f(jmoth)*(b2+a2) + fprev*(a1-b1));

            obj.a1 = a1
            obj.a2 = a2
            obj.b1 = b1
            obj.b2 = b2

            obj.s = [0; 0; 0]; % fprev, bprev, Pox
            obj.A = [0, 0, 0; a1 / b2, b1 / b2, 0; (a1 - b1) / b2, 0, b1 / b2];
            obj.b = [1; a2 / b2; 1 + a2 / b2]; % f

        endfunction

        function [Pout, bnext] = step(obj, f)

            fprev = obj.s(1);
            bprev = obj.s(2);
            Pox = obj.s(3);

            bnext = (1/obj.b2)*(f*obj.a2+fprev*obj.a1+bprev*obj.b1);
            Pout = (1/obj.b2)*(Pox*obj.b1 + f*(obj.b2+obj.a2) + fprev*(obj.a1-obj.b1));

            Pox = Pout;
            bprev = bnext;
            fprev = f;

            obj.s(:) = [f,bnext,Pox];
        
            % obj.s(:) = obj.A * obj.s + obj.b * f;
            % Pout = obj.s(3);
            % bnext = obj.s(2);
        endfunction

    endmethods

endclassdef
