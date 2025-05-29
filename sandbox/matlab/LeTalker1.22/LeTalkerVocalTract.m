classdef LeTalkerVocalTract < LeTalkerTract

    properties
        lip
    endproperties

    properties (Access='protected')
        rhoca
    endproperties

    methods

        function obj = LeTalkerVocalTract(a, fs, atten, R, csnd, rho)
            % a - cross-sectional areas
            % atten - vocal tract attenuation factor (default: 5e-3)

            if nargin < 3
                atten = 0; % use base class default
            end

            if nargin < 4
                R = 0; % use lip class default
            end

            if nargin < 5
                csnd = 35000;
            end

            if nargin < 6
                rho = 0.00114;
            end

            obj@LeTalkerTract(a, atten);
            obj.lip = LeTalkerLips(a(end), fs, R, csnd);

            obj.rhoca = rho * csnd / a(1);
        endfunction

        function [Pout,f_eplx, b_eplx] = step(obj, ug)
            % f1 - forward input pressure
            % b2 - backward input pressure
            %
            % f2 - forward output pressure
            % b1 - backward output pressure

            bprev = obj.get_b_last();
            fprev = obj.get_f_last();

            f1 = obj.alph_odd(1) * obj.b_odd(1) + ug * obj.rhoca;
            [f, b] = obj.step@LeTalkerTract(f1, bprev);

            %------------- Lip Radiation -----------  */
            [Pout, bnext] = obj.lip.step(f);
            obj.set_b_last(bnext);

            f_eplx = f1;
            b_eplx = bnext;
        endfunction

    endmethods

endclassdef
