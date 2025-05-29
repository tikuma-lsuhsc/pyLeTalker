classdef LeTalkerVFFlow<handle


properties (Access='protected')
    a
    csnd
    rhoc
    rhoc_aeplx
    rhoc_atrch    
endproperties

methods

    function obj = LeTalkerVFFlow(Ae, At, rho, csnd)
        % a - between-lip cross-sectional area
        % atten - vocal tract attenuation factor (default: 5e-3)

        if nargin < 3
            rho = 0.00114;
        endif
        if nargin < 4
            csnd = 35000;
        endif

        obj.a = zeros(45);
        obj.a(1) = Ae;
        obj.a(end) = At;
        obj.csnd = csnd;
        obj.rhoc = csnd * rho;
        obj.rhoc_aeplx = rho*csnd/Ae;
        obj.rhoc_atrch = rho*csnd/At;

    endfunction

    function [feplx, btrch, ug] = step(obj, ftrch, beplx, ga)

        % Calculate the glottal flow;
        ug = calcflow(ga, ftrch, beplx, obj.a, obj.csnd, obj.rhoc);
        feplx = beplx + ug * obj.rhoc_aeplx;
        btrch = ftrch - ug * obj.rhoc_atrch;

    endfunction

endmethods

endclassdef



    
    
  
    
