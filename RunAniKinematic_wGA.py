from matplotlib import pyplot as plt
import numpy as np

from KinematicSource import KinematicSource



ANI = 1 #or set to zero to just generate signals
N = 1800 #Number of points in simulation
F0 = 100 #Fund. freq.
x02 = 0.1 #Adduction setting - dist. of superior part of fold from midline
xib = 0.1 #bulging set usually to something between 0.0 and 0.15
npr = 0.7 #nodal point ratio
PL = 8000 #resp pressure
L0 = 1.6 #prephontory VF length
T0 = 0.3 #prephontory VF thickness
F0n = 125 #F0 for normalization
Fs = 44100


#Mesh discretization
Ny = 21
Nz = 15
dt = 1/Fs
#theta = 0

for iter in range(1,N+1):
    
    # x02 = x02tm[iter]
    #     L0 = L0tm[iter]
    #     T0 = T0tm[iter]
    #     F0 = F0tm[iter]
    #     F0n = F0ntm[iter]
    #xib = xibtm[iter]
    [ga[iter],xi,xipos,xi0,y,z,ca[iter],npout,xtest] = KinematicSource(F0,L0,T0,PL,F0n,x02,npr,xib,iter,dt,Ny,Nz)
    
    D[iter] = xtest
    
    S[iter].xi = xipos
    xi0[xi0<0] = 0
    #S[iter].xi = xi0



if (ANI == 1):
    
    Z = np.ones((Ny,Nz))
    Y = np.ones((Ny,Nz))
    
    for i in range(1,Ny+1):
        Z[i,:] = z
    
    
    for i in range(1,Nz+1):
        Y[:,i] = y.T
    
    
    tm = np.arange(N)/Fs
    
    plt.figure()
    
    
    k = 1
    for i in range(1,len(S)+1,12):
        
        plt.surf(Y,S[i].xi,Z)
        plt.colormap(pink)
        #colormap(jet)
        plt.hold
        plt.surf(Y,-S[i].xi,Z)
        #axis([ 0 y()*1.1  -.85  0.80 0 z()*1.1  ])
        plt.axis([ 0, y()*1.1,  -.45,  0.45, 0, z()*1.1  ])
        #axis([ 0 1.1  -.85  0.80 0 0.5])
        #set(gca,'Visible','off')
        
        plt.set(plt.gca,'FontSize',14)
        plt.xlabel('Post.-Ant. (y - cm)','Rotation',25)
        plt.ylabel('Lateral-Medial (x - cm)','Rotation',-20)
        plt.zlabel('Inf.-Sup. (z - cm)')
        
        
        #view([82.5000 -4.0000])
        #view([-37.5000   44.0000])
        plt.view([-50,40])
        #view(-90,90)
        # view([ 63.5000   -2.0000])
        # view([ -68.5000    6.000])
        #view([64.5000  -38.0000])
        
        if(i>1):
            plt.axes('position',[.12, .85, .7, .1])
            set(plt.gca,'FontSize',12)
            plt.plot(tm[1:i:5],ga[1:i:5],'-r','LineWidth',3)
            plt.xlabel('Time (s)','FontSize',12)
            plt.ylabel('A_g (cm^2)','FontSize',12)
            plt.axis([0 N/Fs 0 max(ga)*1.1])
        else:
            plt.axes('position',[.12, .85, .7, .1])
        
        
        
        
        
        M(k) = getframe(gcf)
        k = k+1
        #hold
        clf
        
    
    
    
    
