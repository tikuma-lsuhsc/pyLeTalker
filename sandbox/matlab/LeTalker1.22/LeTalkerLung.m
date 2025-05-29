classdef LeTalkerLung < handle

    properties (WriteAccess='protected')
        PL# vector of lung pressure, if exhausted sustains the last sample
        n1# length of PL vector
        beta# backward pressure reflection loss (0 to let backward pressure to vanish)
    endproperties

    methods

        function obj = LeTalkerLung(PL, beta, tcos, Fs)
            
            if nargin < 2
                beta = 0;
            endif

            if nargin < 3
                tcos = 0.002;
            endif

            if nargin < 4
                Fs = 44100;
            endif

            obj.beta = beta;
            obj.n1 = length(PL);

            ncos = tcos * Fs;
            wcos = pi / ncos;
            ncos = round(ncos);
            n = (0:(ncos-1)).';
            softon = (cos(wcos * n - pi)+ 1) / 2;

            if obj.n1<ncos
                PL = [PL(:); PL(end)*ones(ncos-obj.n1,1)];
                obj.n1 = ncos;
            endif

            PL(1:ncos) = PL(1:ncos) .* softon;
           
            obj.PL = PL;
        endfunction

        function ps = get_PL(obj, n)
            if nargin<2
                ps = obj.PL
            elseif n<=obj.n1
                ps = obj.PL(1:n);
            else
                ps = [obj.PL;obj.PL(end)*ones(n-obj.n1,1)];
            endif
        endfunction

        function f = step(obj, n, b)
            
            % PL = obj.PL
            if n > obj.n1
                PL = obj.PL(end);
            else
                PL = obj.PL(n);
            endif
            
            f = PL - obj.beta * b;
            
        endfunction

    endmethods

endclassdef
