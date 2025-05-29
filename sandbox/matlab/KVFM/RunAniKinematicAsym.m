
F0l = 110; %Fund. freq.
x02l = 0.1; %Adduction setting - dist. of superior part of fold from midline
xibl = 0.25; %bulging set to either 0.0 or 0.15
F0r = 110; %Fund. freq.
x02r = 0.4; %Adduction setting - dist. of superior part of fold from midline
xibr = 0.0; %bulging set to either 0.0 or 0.15
phi0l = 0;
phi0r = 0;
Fs = 44100;
N = 2000;

dt = 1/Fs;
%theta = 0;
clear S ga 
for iter = 1:N
    [ga(iter),xir,xil,y,z] = KinematicSourceAsym(F0r,F0l,x02r,x02l,xibr,xibl,phi0r,phi0l,iter,dt);
    
    
    for i=1:21
        for j=1:15
            if(xir(i,j)-xil(i,j) < 0)
                if(abs(xir(i,j)) < abs(xil(i,j)))
                    xil(i,j) = xir(i,j);
                elseif(abs(xil(i,j)) < abs(xir(i,j)))
                     xir(i,j) = xil(i,j);
                end
            end
        end
    end
    
    
    S(iter).xir = xir;
    S(iter).xil = xil;
    
    
end


figure(1)
clf

Z = ones(21,15);
Y = ones(21,15);

    for i=1:21
        Z(i,:) = z;
    end

    for i=1:15
        Y(:,i) = y';
    end

    tm = [0:1/Fs:(N-1)/Fs];


     figure(1)
    clf

    %hfig = figure(1);
    %set(hfig, 'PaperPositionMode', 'auto');
    k = 1;
    clear M
for i=1:5:length(S)
   
    
    surf(Y,S(i).xir,Z);
    hold
    surf(Y,S(i).xil,Z);
    colormap(pink)
    %axis([ 0 15  0 25.  -.60  0.60]);
    %view([82.5000 -4.0000])
    %view([70.5000  -48.0000])
   % view([ 63.5000   -2.0000]);
   % view([ -68.5000    6.000]);
    %view([64.5000  -38.0000]);
    axis([ 0 y(end)*1.1  -.85  0.80 0 z(end)*1.1  ]);
        %axis([ 0 1.1  -.85  0.80 0 0.5]);
        %set(gca,'Visible','off')
        
        set(gca,'FontSize',14);
        xlabel('Post.-Ant. (y - cm)','Rotation',25)
        ylabel('Lateral-Medial (x - cm)','Rotation',-20);
        zlabel('Inf.-Sup. (z - cm)');


        %view([82.5000 -4.0000])
        %view([-37.5000   44.0000])
        view([-50 40])
        
        
         if(i>1)
            axes('position',[.12 .85 .7 .1])
            set(gca,'FontSize',12)
            plot(tm(1:5:i),ga(1:5:i),'LineWidth',2)
            xlabel('Time (s)','FontSize',12)
            ylabel('A_g (cm^2)','FontSize',12);
            axis([0 N/Fs 0 max(ga)*1.05])
        else
            axes('position',[.12 .85 .7 .1])
        end
        

    M(k) = getframe(gcf);
    k = k+1;
    clf
    %hold
end


