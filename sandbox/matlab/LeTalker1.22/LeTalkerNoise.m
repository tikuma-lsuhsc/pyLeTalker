classdef LeTalkerNoise < handle

    properties (WriteAccess='protected')
        rho_Lmu
        RE2b
        atten
        cb
        ca
        xnois
        ynois
    endproperties

    methods

        function obj = LeTalkerNoise(L, REc, atten, cb, ca, rho, mu)

            if nargin < 2
                REc = 1200;
            endif

            if nargin < 3
                atten = 4e-6 / sqrt(12); % uniform -> normal std adjustment
            endif

            if nargin < 4
                cb = [0.0289, 0, -0.0578, 0, 0.0289];
            endif

            if nargin < 5
                ca = [1, -3.4331, 4.4547, -2.602, 0.5806];
            endif

            if nargin < 6
                rho = 0.00114;
            endif

            if nargin < 7
                mu = 0.000186;
            endif

            obj.RE2b = REc^2;
            obj.atten = atten;
            obj.rho_Lmu = rho / (L * mu);

            # IIR filter (Direct Form I) coefficients as row vectors
            obj.cb = cb(:).';
            ca = ca(2:end);
            obj.ca = ca(:).';

            # filter states as column vectors
            obj.xnois = zeros(length(obj.cb), 1);
            obj.ynois = zeros(length(obj.ca), 1);

        endfunction

        function Unois = step(obj, ug)

            % update the input tap line with a new random sample
            obj.xnois(2:end) = obj.xnois(1:end - 1);
            % obj.xnois(1) = rand(1) - 0.5; # uniform distribution: var: 1/12
            obj.xnois(1) = randn(1); % Gaussian/normal distribution with var = 1

            % filter operation
            out = obj.cb * obj.xnois - obj.ca * obj.ynois;

            # if Reynold's number is above threshold scale noise and output
            RE2 = (ug * obj.rho_Lmu)^2;

            if RE2 > obj.RE2b
                Unois = obj.atten * (RE2 - obj.RE2b) * out;
            else
                Unois = 0;
            end

            % update the output feedback tap line with the latest output sample
            obj.ynois(2:end) = obj.ynois(1:end - 1);
            obj.ynois(1) = out;

        endfunction

    endmethods

endclassdef
