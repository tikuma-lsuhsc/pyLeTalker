classdef LeTalkerTrachea < LeTalkerTract

    properties (SetAccess='private', GetAccess='public')
    endproperties

    properties (Access='protected')
        rhoca
    endproperties

    methods

        function obj = LeTalkerTrachea(a, atten, csnd, rho)
            % a - cross-sectional areas
            % atten - vocal tract attenuation factor (default: 5e-3)

            if nargin < 2
                atten = 0; % use base class default
            end

            if nargin < 3
                csnd = 35000;
            end

            if nargin < 4
                rho = 0.00114;
            end

            obj@LeTalkerTract(a, atten);

            obj.rhoca = rho * csnd / a(end);

        endfunction

        function [psg, b1] = step(obj, ps, ug)
            % ps - lung pressure
            % ug - transglottal flow
            %
            % psg - subglottal pressure

            f1 = 0.9 * ps - 0.8 * obj.b_odd(1) * obj.alph_odd(1);
            alpha = obj.get_alpha_last();
            brefl = obj.get_f_last() * alpha;
            pg = ug * obj.rhoca;
            b2 = brefl - pg / alpha;
            psg = 2 * brefl - pg;

            [f2, b1] = obj.step@LeTalkerTract(f1, b2);
            

        endfunction

    endmethods

endclassdef
