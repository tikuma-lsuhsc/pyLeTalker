function [fo,tme] = zerocross(x,Fs);
%
Dt = 1/Fs; 
t = [0:Dt:(length(x)-1)*Dt];
ind = 1;
tme = [];
for i = 1:length(x)-1
  if(x(i) < 0 && x(i+1) > 0)
    dt = -(Dt*x(i))/(x(i+1) - x(i));
    tme(ind) = t(i)+dt;
          ind = ind+1;
        
  end;
end;
tme;

if(length(tme) >1)
    fo = 1./diff(tme)';
    fo = [fo(1); fo; fo(length(fo))];
    tme = [0.0; tme'];
else
    fo = 0;
end



