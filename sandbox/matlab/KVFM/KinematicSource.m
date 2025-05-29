function [ga,xi,xipos,xi0,y,z,ca,npout,xtest] = KinematicSource(F0,L0,T0,PL,F0n,x02,np,xib,iter,dt,Ny,Nz);

goffset = 0.0001;
%------test-------
%freq = p.freq;
%freqp = p.freqp;
freq = F0;
%F0n = 125;
%F0n = 270;
%freqp = 105;

%L0 = 1.6;
%T0 = 0.3;
%PL = 4000;
L = 0.56*L0*sqrt((F0/F0n));
eps = (L - L0)/L0;
T = T0/(1+0.8*eps);
%T = 0.5;
zn = np*T;
pth = 10000*(0.14 + .06*(F0/F0n)^2 );

if(PL>pth)
    xm = 0.05*L0*sqrt((PL-pth)/pth);
else
    xm = 0;
end


c =pi*F0*T;

%xcv = 0.4;
xcv =  (4.0-4.4*zn/T)/10; 
phi0 = 0;

%x02 = 0.4;
pgap = 0;
x01 = xcv+x02;


xi02 = x02;
xi01 = x01;

%attenuation with abduction - ad hoc rule
xm = 0.1;
xm = xm * max(0,0.25* (cos(x02*3.5)+1)^2 ); 
    


%----normalized time-----------
% if(iter == 1)
%     theta = 0;
% else
%     theta = theta + pi*(freq + freqp)*dt;
% end
% 
% if(theta >= 2*pi)
%     theta = theta - 2*pi;
% end

flag = 9999999;
t = iter*dt;
omega = 2*pi*freq;
%t = theta/omega;

ny = Ny;
nz = Nz;
dydz = (L/(ny-1)) * (T/(nz-1));
ga = 0;
ca = 0;

y = [0:ny-1]*L/(ny-1);
z = [0:nz-1]*T/(nz-1);
xi1 = zeros(ny,nz);
xi = zeros(ny,nz);
xipos = zeros(ny,nz);
xineg = zeros(ny,nz);


sth = sin(omega*t+phi0);
cth = cos(omega*t+phi0);

for i=1:ny
    lengtri = 1-y(i)/L;
    amplsin = xm*sin(pi*y(i)/L);
    
    for j=1:nz
        xi0(i,j) = lengtri*(x02 + (x01 - x02 - 4*xib*z(j)/T)*(1-z(j)/T) )+goffset;
        %xi1(i,j) = amplsin*sin(omega*(t-z(j)/c) + phi0);
        xi1(i,j) = amplsin*(sth - omega*(z(j)-zn)*cth/c); %linearized
        
        
        if(abs(z(j)-zn) < flag)
            flag = abs(z(j)-zn);
            npout = j;
        end
        
        %version
        
        %add nodule----------------
%          if(j>=6 & j<=15 & i>=10 & i<=15)
%              xi0(i,j) = xi0(i,j) - .12;
%          end
        
        
        %-------------------
        
        xi(i,j) = xi0(i,j) + xi1(i,j);
        xipos(i,j) = max(0,xi(i,j));
        xineg(i,j) = min(0,xi(i,j));
        
        if(i==10 && j==8)
            %xtest = xi(i,j);
            xtest = xi1(i,j);
        end
        
    end
end

%glottal area calculation
for i=1:ny-1
    g = xipos(i,1) + xipos(i+1,1); %initial area estimate
    
     for j=1:nz-1
         g = min(g, xipos(i,j+1) + xipos(i+1,j+1) );
     end
     
     ga = ga + g;
end

ga = ga*L/(ny-1);
ga = ga + pgap;



%Contact area 

for i=1:ny-1
    
    xminus = 0;
    xmag = 0;
     for j=1:nz-1
         
         %Attempt at Druker's version
         %xminus = xminus + -(min(0,xi(i,j)) + min(0,xi(i+1,j)) + min(0,xi(i,j+1)) + min(0,xi(i+1,j+1)));
        % xmag = xmag + abs(xi(i,j)) + abs(xi(i+1,j)) + abs(xi(i,j+1)) + abs(xi(i+1,j+1));
         
        %simply add up the number of  elements that have crossed the
        %midline
         xminus = xminus + ( (min(0,xi(i,j))<0) + (min(0,xi(i+1,j))<0) + (min(0,xi(i,j+1))<0) + (min(0,xi(i+1,j+1)) <0) );
         
     end
     
     %divide by 4 because each set of four elements constructs a square; 
     %it is the number of squares that needs to be multiplied by dydz 
     ca = ca + dydz*xminus/4; 
     
end


        