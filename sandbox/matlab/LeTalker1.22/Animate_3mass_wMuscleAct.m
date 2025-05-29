function M = Animate_3mass_wMuscleAct(r,ctvect,tavect,skipf)
%Author: Brad Story, University of Arizona
%Last updated on 10.05.2012
%
%Code to Animate the 3 mass model
%INPUTS:
%r = structure of waveforms resulting from running LeTalkerGUI
%skipf = number of samples to skip for the animation; must be equal to 1 or
%greater. It is recommended that skipf be set to at least 10, otherwise the
%animation will be really slow.
%OUTPUTS:
%M = matlab movie structure that can be written to various movie formats
%(e.g., mpgwrite.m)
%NOTE: LeTalker produces a bilaterally symmetric vibration pattern.The left
%and right sides shown in the animation are simply mirrors of each other.

if(skipf < 1)
    skipf = 1;
    disp('Reset skipf = 1');
end

figure(5);
clf
Xu = r.x2;
Xl = r.x1;
Xb = r.xb;
%hold;

Xu = Xu(1:skipf:end);
Xl = Xl(1:skipf:end);
Xb = Xb(1:skipf:end);
ug = r.ug(1:skipf:end);
ctvect = ctvect(1:skipf:end);
tavect = tavect(1:skipf:end);


yu = [0 .15 .15 0];
x = [0 0 .15 .15];
xb = [.35 .35 .55 .55];
yb = [-.15 .15 .15 -.15];
yl = [0 -.15 -.15 0];

for i=1:length(Xu)
    
    
    if(Xl(i) < 0) Xl(i) = 0;
    end
    
    if(Xu(i) < 0) Xu(i) = 0;
    end
    
    fill(Xl(i)+x,yl,[.5 .6 .6]);
    hold;
    fill(Xu(i)+x,yu,[.5 .6 .7]);
    fill(-Xl(i)-x,yl,[.5 .6 .6]);
    fill(-Xu(i)-x,yu,[.5 .6 .7]);
    
    fill(Xb(i)+xb,yb,[.5 .5 .5]);
    fill(-Xb(i)-xb,yb,[.5 .5 .5]);
    
    
    plot([Xl(i)+x(3) Xb(i)+xb(1)],[yl(2)/2 yl(2)/2],'k','LineWidth',2);
    plot([Xu(i)+x(3) Xb(i)+xb(1)],[yu(2)/2 yu(2)/2],'k','LineWidth',2);
    plot([Xb(i)+xb(3) .75],[0 0],'k','LineWidth',2);
    plot([.75 .75],[-.2 .2],'k','LineWidth',2);
    
    
    plot([-Xl(i)-x(3) -Xb(i)-xb(1)],[yl(2)/2 yl(2)/2],'k','LineWidth',2);
    plot([-Xu(i)-x(3) -Xb(i)-xb(1)],[yu(2)/2 yu(2)/2],'k','LineWidth',2);
    plot(-[Xb(i)+xb(3) .75],[0 0],'k','LineWidth',2);
    plot(-[.75 .75],[-.2 .2],'k','LineWidth',2);
    
    
    plot([Xl(i) xb(1)],[yl(2) yl(2)-.2],'k','LineWidth',2);
    plot([Xu(i) xb(1)],[yu(2) yu(2)+.1],'k','LineWidth',2);
    plot(-[Xl(i) xb(1)],[yl(2) yl(2)-.2],'k','LineWidth',2);
    plot(-[Xu(i) xb(1)],[yu(2) yu(2)+.1],'k','LineWidth',2);
    
    
    plot([xb(1) xb(1)],[yl(2)-.2 yl(2)-.5],'k','LineWidth',2);
    plot(-[xb(1) xb(1)],[yl(2)-.2 yl(2)-.5],'k','LineWidth',2);
    plot([xb(1) xb(1)],[yu(2)+.1 yu(2)+.5],'k','LineWidth',2);
    plot(-[xb(1) xb(1)],[yu(2)+.1 yu(2)+.5],'k','LineWidth',2);
    
%        if(Xl(i) <= 0 )
%            xtmp = [Xl(i) xb(1)/3 xb(1)/3 -xb(1)/3 -xb(1)/3 -Xl(i)];
%            ytmp = [yl(2) yl(2)-.2 yl(2)-.5 yl(2)-.5 yl(2)-.2 yl(2)];
%            fill(xtmp,ytmp,'r');
%        elseif(Xl(i) > 0 && Xu(i) <=0)
%            xtmp = [Xl(i) Xl(i) xb(1)/3 xb(1)/3 -xb(1)/3 -xb(1)/3 -Xl(i) -Xl(i)];
%            ytmp = [0 yl(2) yl(2)-.2 yl(2)-.5 yl(2)-.5 yl(2)-.2 yl(2) 0];
%            fill(xtmp,ytmp,'r');
%        elseif(Xl(i) > 0 && Xu(i) > 0)
%            xtmp = [xb(1)/3   xb(1)/3  Xu(i)  Xu(i) Xl(i) Xl(i)  xb(1)/3   xb(1)/3   -xb(1)/3    -xb(1)/3 -Xl(i) -Xl(i) -Xu(i) -Xu(i)  -xb(1)/3 -xb(1)/3  ];
%            ytmp = [yu(2)+.5  yu(2)+.1 yu(2)   0     0     yl(2) yl(2)-.2  yl(2)-.5   yl(2)-.5   yl(2)-.2  yl(2)  0        0     yu(2)   yu(2)+.1  yu(2)+.5];
%            fill(xtmp,ytmp,'r');
%     
%        end
    
    
    
    %set(gca,'PlotBoxAspectRatio',[.6 1 1])
    set(gca,'PlotBoxAspectRatio',[1 1 1])
    
    %    plot(Xl(i), -.075,'ob');
    %    hold
    %    plot(Xu(i), .075,'ob');
    %    plot(-Xl(i), -.075,'or');
    %    plot(-Xu(i), .075,'or');
    %
    %    plot(Xb(i) + .1, 0,'ob');
    %    plot(-Xb(i) - .1, 0,'ob');
    
    %text(-.1,-.3, 'Trachea');
    %text(-.1,.3, 'Vocal Tract');
    
    h=text(-.05,-.4, 'Trachea');
    set(h,'FontSize',14,'FontWeight','bold');
    h = text(-.05,.4, 'Vocal Tract');
    set(h,'FontSize',14,'FontWeight','bold');
    
    h = text(-0.05,0.3,['U_g = ' num2str(  round(ug(i)) ) ]);
    set(h,'FontSize',14,'FontWeight','bold');
    
    h = text(-0.65,0.3,['a_{CT}= ' num2str( round(100*ctvect(i))/100)]);
    set(h,'FontSize',14,'FontWeight','bold','Color',[1 0 0]);
    h = text(-0.65,0.25,['a_{TA}= ' num2str( round(100*tavect(i))/100)]);
    set(h,'FontSize',14,'FontWeight','bold','Color',[0 0 1]);
    
    
    h = text(-0.8,-0.45,'B. Story, U. Arizona, 2013');
    set(h,'FontSize',12,'FontWeight','bold');
    
    axis([-.8 .8 -.6 .6]);
    set(gca,'Visible','off');
    set(gcf,'Color',[1 1 1]);
    M(:,i) = getframe;
    hold
end;
