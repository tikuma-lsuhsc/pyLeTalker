function [r,h,f,hrc,frc] = LeTalkerImpResp(p,c,Fs,flag)
%LeTalkerImpResponse- calculate the impulse response of the vocal tract
%area function
%LeTalker - means "Lumped element Talker" and is the control code
%for running the three-mass model of Story and Titze (1995) and Titze and Story (2002) with sub and supra glottal
% airways;
%INPUTS: p and c are structures the result from running LeTalker or
%LeTalkerGUI, Fs = sampling freq, usually 44100 Hz, flag = 0 will run
%simulation with lip radiation impedance, flag =1 runs will ideal end
%condition (complete reflection).
%OUTPUTS: r = structure of signals, h = vocal tract freq response based on
%impulse response, f = freq. vector for h; hrc = vocal tract transfer
%function based r2 and rc2poly, frc = freq vector.
% Author: Brad Story, Univ. of Arizona
%Date: 06.18.2014

%----Set N to be desired size of FFT----
N = 4096; 

%-----
a = p.ar(1,1:44);
Nsect = 44;
ieplx = 1;
jmoth = 44;

f = 	zeros(1,Nsect);
b = 	zeros(1,Nsect);
r1 = 	zeros(1,Nsect);
r2 = 	zeros(1,Nsect);


r.po = []; %output pressure
r.ug = []; %glottal flow


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

%vocal tract attenuation - based on xsect area
alpha = 1-p.vtatten./sqrt(a);

for n=1:N
    t = (n-1)*dt;
   
    % Set the glottal flow to be an impulse;
   
    if(n==200) 
        ug = 1;
    else
        ug = 0;
    end
    
    
    %   ============== wave propagation in vocal tract ======================= */
    
    D = a(1:jmoth-1) + a(2:jmoth);
    r1 = (a(1:jmoth-1) - a(2:jmoth))./D;
    r2 = -r1;
    
    
    % ---even sections in trachea--- */
    
    f(ieplx) = alpha(ieplx)*b(ieplx) + ug*(rhoc/a(ieplx));
 
    f  = alpha.*f;
    b = alpha.*b;
    
    % ---even sections in supraglottal--- */
    
    Psi = f(ieplx+1:2:jmoth-2).*r1(ieplx+1:2:jmoth-2)  + b(ieplx+2:2:jmoth-1).*r2(ieplx+1:2:jmoth-2);
    b(ieplx+1:2:jmoth-2) = b(ieplx+2:2:jmoth-1) + Psi;
    f(ieplx+2:2:jmoth-1) = f(ieplx+1:2:jmoth-2) + Psi;
    
    
    % odd sections */
    
    Psi = f(ieplx:2:jmoth-1).*r1(ieplx:2:jmoth-1)  + b(ieplx+1:2:jmoth).*r2(ieplx:2:jmoth-1);
    b(ieplx:2:jmoth-1) = b(ieplx+1:2:jmoth) + Psi;
    f(ieplx+1:2:jmoth) = f(ieplx:2:jmoth-1) + Psi;
    
    
    %------------- Lip Radiation -----------  */
    am = sqrt(a(jmoth)/PI);  %am is the radius of the mouth opening
    L = (2.0/dt)*8.0*am/(3.0*PI*csnd);
    a2 = -R - L + R*L;
    a1 = -R + L - R*L;
    b2 = R + L + R*L;
    b1 = -R + L + R*L;
    b(jmoth) = (1/b2)*(f(jmoth)*a2+fprev*a1+bprev*b1);
    Pout = (1/b2)*(Pox*b1 + f(jmoth)*(b2+a2) + fprev*(a1-b1));
    Pox = Pout;
    
    bprev = b(jmoth);
    fprev = f(jmoth);

    
    % --Complete reflection at lips------
    if(flag == 1)
      b(jmoth) = -f(jmoth);
      Pout = f(jmoth);
    end
    %---------------------------------------- */
    
    tme  = tme +dt;
    
    %Assign output signals to structure--------
    r.po(n) = Pout;
    r.pi(n) = f(1) + b(1);
    r.ug(n) = ug;
  
end;

%---Calcuate Frequency Response based on impulse-----

h = fft(r.po,N);
f  = [0:Fs/N:(N-1)*(Fs/N)];
figure(2);
clf
hold on
plot(f,20*log10(abs(h)),'-b');
axis([0 5000 -30 70]);

% %----Calculate Input Impedance-----
% h = fft(r.pi,N);
% f  = [0:Fs/N:(N-1)*(Fs/N)];
% 
% plot(f,20*log10(abs(h)),'-r');
% axis([0 5000 -30 70]);

%----Calculate Frequency Response based on rc2poly.m
r2(44) = 1;  %Need an ideal end condition
z = rc2poly(r2);
[hrc,frc] = freqz(1,z,2000,44100);
plot(frc,20*log10(abs(hrc)),'-k');

set(gca,'FontSize',14);
if(flag == 0)
    title('Radiation Impedance ON');
else
    title('NO radiation impedance - full reflection at lips');
end

legend('Freq Response - Impulse','Freq Response - ref. coeffs');
xlabel('Frequency (Hz)');
ylabel('Rel. Ampl. (dB)');
grid


