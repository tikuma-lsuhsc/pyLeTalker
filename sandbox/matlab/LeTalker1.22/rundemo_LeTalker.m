%Run Code as Demonstration
%B. Story
%10.25.11

function [p, r, Fs] = rundemo_LeTalker(N, vt_data, tr_data)
    Fs = 44100;
    InitializeLeTalker

    if isnan(vt_data)
    else
        p.ar(1:44) = vt_data;
    end

    if isnan(tr_data)
    else
        p.ar(45:76) = tr_data;
    end

    p.SUPRA = 1;
    p.SUB = 1;
    p.Noise = 1;

    rand ("seed", 0);

    [p, r] = LeTalker(p, c, N, Fs, [], [], []);
    % [p1, r1] = LeTalker1(p, c, N, Fs, [], [], []);

    % p = r.po'
    % r = r1.po
end

% PlotLeTalkerWaveforms(r,p,Fs,1,2);
