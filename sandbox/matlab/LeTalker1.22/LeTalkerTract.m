classdef LeTalkerTract < handle

    properties (SetAccess='private', GetAccess='public')
        a% cross-sectional area from source end
        atten% vocal tract attenuation factor
    endproperties

    properties (Access='protected')
        r_odd
        r_even

        even_end
        alph_odd
        alph_even
        f_odd
        f_even
        b_odd
        b_even

    endproperties

    methods

        function obj = LeTalkerTract(a, atten)
            % a - cross-sectional areas
            % atten - vocal tract attenuation factor (default: 5e-3)

            if nargin < 2 || atten <= 0
                atten = 5e-3;
            endif

            obj.a = a(:);
            obj.atten = atten;
            D = obj.a(1:end - 1) + obj.a(2:end);
            r1 = (obj.a(1:end - 1) - obj.a(2:end)) ./ D;
            obj.r_odd = r1(1:2:end);
            obj.r_even = r1(2:2:end);

            alpha = 1 - atten ./ sqrt(obj.a);
            obj.alph_odd = alpha(1:2:end);
            obj.alph_even = alpha(2:2:end);

            n = length(a);
            obj.even_end = mod(n, 2) == 0;
            nodd = floor(n / 2);
            neven = n - nodd;
            obj.f_odd = zeros(nodd, 1); % [F1,F3,...]
            obj.f_even = zeros(neven, 1); % [F2,F4, ...]
            obj.b_odd = zeros(nodd, 1); % [B1,B3,...]
            obj.b_even = zeros(neven, 1); % [B2,B4, ...]

        endfunction

        function [f2, b1] = step(obj, f1, b2)
            % f1 - forward input pressure
            % b2 - backward input pressure
            %
            % f2 - forward output pressure
            % b1 - backward output pressure

            % ---even sections--- */

            % --- feed the new inputs
            obj.f_odd(1) = f1;
            obj.set_b_last(b2);

            % --- attenuate pressure due to tract loss factor
            b_odd = obj.b_odd .* obj.alph_odd;
            f_odd = obj.f_odd .* obj.alph_odd;
            b_even = obj.b_even .* obj.alph_even;
            f_even = obj.f_even .* obj.alph_even;
            ioffset = obj.even_end;

            % ---even junctions [(F2,B3),(F4,B5),...]->[(F3,B2),(F5,B4),...]--- */
            Psi = (f_even(1:end - ioffset) - b_odd(2:end)) .* obj.r_even;
            f_odd(2:end) = f_even(1:end - ioffset) + Psi;
            b_even(1:end - ioffset) = b_odd(2:end) + Psi;

            % ---odd junctions [(F1,B2),(F3,B4),...]->[(F2,B1),(F4,B3),...]--- */
            ioffset =~ioffset;
            Psi = (f_odd(1:end - ioffset) - b_even) .* obj.r_odd;
            f_even(:) = f_odd(1:end - ioffset) + Psi;
            b_odd(1:end - ioffset) = b_even + Psi;

            % Psi = f(2:2:jmoth-2).*r1(2:2:jmoth-2)  + b(3:2:jmoth-1).*r2(2:2:jmoth-2);
            % f(3:2:jmoth-1) = f(2:2:jmoth-2) + Psi;
            % b(2:2:jmoth-2) = b(3:2:jmoth-1) + Psi;
            % Psi = f(1:2:jmoth-1).*r1(1:2:jmoth-1)  + b(2:2:jmoth).*r2(1:2:jmoth-1);
            % f(2:2:jmoth) = f(1:2:jmoth-1) + Psi;
            % b(1:2:jmoth-1) = b(2:2:jmoth) + Psi;

            obj.b_odd(:) = b_odd;
            obj.f_odd(:) = f_odd;
            obj.b_even(:) = b_even;
            obj.f_even(:) = f_even;

            f2 = obj.get_f_last();
            b1 = b_odd(1);
        endfunction

        function v = get_alpha_first(obj)
            v = obj.alph_odd(1);
        endfunction

        function v = get_alpha_last(obj)

            if obj.even_end
                v = obj.alph_even(end);
            else
                v = obj.alph_odd(end);
            endif

        endfunction

        function v = get_f_first(obj)
            v = obj.f_odd(1);
        endfunction

        function set_f_first(obj, v)
            obj.f_odd(1) = v;
        endfunction

        function v = get_b_first(obj)
            v = obj.b_odd(1);
        endfunction

        function set_b_first(obj, v)
            obj.b_odd(1) = v;
        endfunction

        function v = get_f_last(obj)

            if obj.even_end
                v = obj.f_even(end);
            else
                v = obj.f_odd(end);
            endif

        endfunction

        function set_f_last(obj, v)

            if obj.even_end
                obj.f_even(end) = v;
            else
                obj.f_odd(end) = v;
            endif

        endfunction

        function v = get_b_last(obj)

            if obj.even_end
                v = obj.b_even(end);
            else
                v = obj.b_odd(end);
            endif

        endfunction

        function set_b_last(obj, v)

            if obj.even_end
                obj.b_even(end) = v;
            else
                obj.b_odd(end) = v;
            endif

        endfunction

    endmethods

endclassdef
