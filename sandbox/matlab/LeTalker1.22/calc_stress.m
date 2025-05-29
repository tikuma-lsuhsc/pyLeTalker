function p = calc_stress(p)
% Stress calculations for the three-mass model based on Titze and Story
% 2002
% Author: Brad Story

for i=1:3 
   ep1(i) = -.5; strs(i) = 0.0; 
   ep2(i) = 0.0;
end;

%-----------
ep2(1) = -0.35;
ep2(2) = 0.0;
ep2(3) = -0.05;

sig0(1) = 5000.; sig0(2) = 4000; sig0(3) = 10000;
sig2(1) = 300000;  sig2(2) = 13930; sig2(3) = 15000;
Ce(1) = 4.4; Ce(2) = 17; Ce(3) = 6.5;

for i=1:3
  sig_lin = (-sig0(i)/ep1(i))*(p.eps - ep1(i));
  if(p.eps > ep2(i) )
    sig_nln = sig2(i)*( exp(Ce(i)*(p.eps-ep2(i)))-1-Ce(i)*(p.eps-ep2(i)));
  else
    sig_nln = 0.0;
  end;
  
  strs(i) = sig_lin + sig_nln;
end;


p.sigmuc = strs(1);
p.sigl = strs(2);
p.sigp = strs(3);
