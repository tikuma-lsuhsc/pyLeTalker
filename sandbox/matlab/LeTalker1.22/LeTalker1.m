function [p, r] = LeTalker1(p,c,N,Fs,ctvect,tavect,plvect)
%LeTalker version 1.23
%LeTalker - means "Lumped element Talker" and is the control code
%for running the three-mass model of Story and Titze (1995) and Titze and Story (2002) with sub and supra glottal
% airways;
% Author: Brad Story, Univ. of Arizona
%Date: 06.18.2014

p.ar = p.ar(:);

p = rules_consconv(p,c);

%load filter coefficients for band pass filtering glottal noise
load filter_coeffs;

%If SUB=1 the subglottal system is included; SUB=0 will take it out.
SUB = p.SUB;
if(SUB ==0)
    p.ar(end) = 1000000;
end

%If SUPRA=1 the supraglottal system is included; SUPRA=0 will take it out.
SUPRA = p.SUPRA;
if(SUPRA == 0)
    p.ar(1) = 1000000;
end


neq = 6;
F = 	zeros(neq,1);
r.theta = zeros(N,1);
r.x = zeros(N,1);
r.xb = zeros(N,1);

p.tan_th0 = p.xc/p.T;
p.th0 = atan(p.tan_th0);

for i=1:6
    F(i) = 0;
end;

%For use with Titze 1 below-----
F(1) = 0;
F(2) = 0;
F(3) = 0;
%For use with Titze 2 below-----
% F(1) = p.x01;
% F(2) = p.x02;
% F(3) = p.xb0;


%-----
a = p.ar(:);
Nsect = length(a);
ieplx = 1;
jmoth = 44;
itrch = jmoth+1;
jtrch = Nsect;
f = 	zeros(Nsect,1);
b = 	zeros(Nsect,1);
r1 = 	zeros(Nsect,1);
r2 = 	zeros(Nsect,1);
aprev = zeros(jmoth,1);
atot = zeros(jmoth,1);

r.x1 = zeros(N,1); %lower mass disp
r.x2 = zeros(N,1); %upper mass disp
r.xb = zeros(N,1); %body mass disp
r.po = zeros(N,1); %output pressure
r.ug = zeros(N,1); %glottal flow
r.ga = zeros(N,1); %glottal area
r.psg = zeros(N,1);
r.nois = zeros(N,1);
r.ad = zeros(N,1);
r.ps = zeros(N,1);
r.pi = zeros(N,1);
r.f1 = zeros(N,1);
r.f2 = zeros(N,1);

%---Initialize vectors for noise generation--
xnois = zeros(5,1);
ynois = zeros(5,1);
Unois = 0;

%vocal tract attenuation - same everywhere
%alpha = .98*ones(Nsect);
csnd =35000;
rho  =0.00114;
rhoc =rho*csnd;
mu = 0.000186;
PI = pi;
PI2 = 2*PI;

bprev = 0;
fprev = 0;
Pox = 0.0;

dt = 1/Fs;
tme = 0.0;
R = 128.0/(9.0*pi.^2);

%--initialize----------
p.a1 = 2*p.L*F(1);
p.a2 = 2*p.L*F(2);
p.ga = max(0,min(p.a1,p.a2));

%vocal tract attenuation - based on xsect area
alpha = 1-p.vtatten./sqrt(a);

%posterior glottal gap
pgap = 0;
p.psg = p.ps;

%------

tcos = 0.002;

if SUPRA
    vt = LeTalkerVocalTract(p.ar(ieplx:jmoth), Fs);
endif

if SUB
    tr = LeTalkerTrachea(p.ar(itrch:jtrch), p.vtatten);
    lung = LeTalkerLung(0.9 * p.PL, 0.8 * tr.get_alpha_first());
else
    lung = LeTalkerLung(p.PL);
endif

r.ps = lung.get_PL(N)/0.9;

if p.Noise
    ng = LeTalkerNoise(p.L);
endif

b_lung = 0;
ug = 0
f_eplx = 0;
b_eplx = 0;
f_trch = 0;

for n=1:N
    t = (n-1)*dt;
    
    %-- check for time-varying muscle activation and pressure ramp
    if(isempty(tavect)==0)
        p.ata=tavect(n);
    end
    if(isempty(ctvect)==0)
        p.act=ctvect(n);
    end

    p.ps = lung.step(n, b_lung);
    
    if SUB
        % tr.set_b_last(b(jtrch));
        [p.psg, b_lung] = tr.step(p.ps, ug);
        f_trch = tr.get_f_last();
    else
        p.psg = p.ps;
    endif

    %-------------------------
    
    p = rules_consconv(p,c);
    
    %This is internally necessary because p.ps was changed to p.psg in calc_pressures.m to accomodate the trachea wave
    %propagation for other implementations
    % p.psg = p.ps;
  
    
    
    %================Titze 1 ==========================
    
    znot = p.zn/p.T;
    p.xn = (1-znot)*(p.x01+F(1))+znot*(p.x02 + F(2));
    p.tangent = (p.x01-p.x02 + 2*(F(1)-F(2)))/p.T;
    p.x1 = p.xn - ( -p.zn)*p.tangent;
    p.x2 = p.xn - (p.T - p.zn)*p.tangent;
    %====================================================
    
    
    %================Titze 2 ==========================
    %  p.tan_th0 = (p.x01-p.x02)/p.T;
    %  p.th0 = atan(p.tan_th0);
    %  p.tangent = tan(p.th0 + F(1));
    %  p.xn = p.x02 + (p.T - p.zn)*p.tan_th0 + F(2);
    %  p.x1 = p.xn - ( -p.zn)*p.tangent;
    %  p.x2 = p.xn - (p.T - p.zn)*p.tangent;
    %====================================================
    
    % Vocal fold area calculations
    p.zc = min(p.T, max(0., p.zn+p.xn/p.tangent));
    p.a1 = max(p.delta, 2*p.L*p.x1);
    p.a2 = max(p.delta, 2*p.L*p.x2 );
    p.an = max(p.delta, 2*p.L*p.xn);
    p.zd = min(p.T, max(0, -0.2*p.x1/p.tangent));
    p.ad = min(p.a2, 1.2*p.a1);
    
    % contact area calculations
    if(p.a1 > p.delta && p.a2 <= p.delta)
        p.ac = p.L*(p.T - p.zc);
    elseif(p.a1 <= p.delta && p.a2 > p.delta)
        p.ac = p.L*p.zc;
    elseif(p.a1 <= p.delta && p.a2 <= p.delta)
        p.ac = p.L*p.T;
    end;
    
    
    %Calculate driving pressures
    p.pe = f_eplx+b_eplx;
    p = calc_pressures(p.psg,p.pe,p);
    
    %Integrate EOMs with Runge-Kutta
    F = rk3m(neq,F,tme,dt,p);
    
    %Glottal area calculation
    p.ga = max(0,min(p.a1,p.a2));
    
    % Calculate the glottal flow;
    ug = calcflow(p.ga,f_trch,b_eplx,a,csnd,rhoc);
    
    
    %=========noise generator====================
    %Added on 10.04.12 - Reynolds number calculation and option for adding
    %noise to simulate turbulence-generated sound, based on Samlan and
    %Story (2011) JSLHR
    if p.Noise == 1
        Unois = ng.step(ug);
        ug = ug + Unois;
    end
    
    if SUPRA
        [Pout,f_eplx,b_eplx] = vt.step(ug);
    endif

    %---------------------------------------- */
    
    tme  = tme +dt;
    
    
    %Assign output signals to structure--------
    r.po(n) = Pout;
    r.ug(n) = ug;
    r.nois(n) = Unois;
    r.ga(n) = p.ga;
    r.ad(n) = p.ad;
    r.x1(n) = p.x1;
    r.x2(n) = p.x2;
    r.xb(n) = F(3);
    r.psg(n) = p.psg;
    r.pi(n) = p.pe;
    r.f1(n) = p.f1;
    r.f2(n) = p.f2;
    
    
    
    if(max(isnan(r.x1)) == 1 | max(isnan(r.x2)) | max(isnan(r.xb)) )
        r.x1 = zeros(N,1);
        r.x2 = zeros(N,1);
        break;
    elseif(max(isinf(r.x1)) == 1 | max(isinf(r.x2)) | max(isinf(r.xb)) )
        r.x1 = zeros(N,1);
        r.x2 = zeros(N,1);
        break;
    end;
    
    
    
end;

tmp = r.x1;
n1 = round(0.4*length(tmp));
n2 = length(tmp)-round(0.1*length(tmp));
tmp = tmp(n1:n2);
tmp = tmp;
if(min(tmp)>0);
    tmp = tmp-0.1*max(tmp);
end

[f0,tme] = zerocross(tmp,Fs);

r.f0 = round(mean(f0));
