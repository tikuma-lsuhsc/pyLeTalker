function M = Animate_3mass_wquantities(r,p,skipf)
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

figure(1);
clf
Xu = r.x2;
Xl = r.x1;
Xb = r.xb;
%hold;

Xu = Xu(1:skipf:end);
Xl = Xl(1:skipf:end);
Xb = Xb(1:skipf:end);
Pi = r.pi(1:skipf:end);
Ps = r.ps(1:skipf:end);
a2 = Xu*p.L;
a1 = Xl*p.L;
f1 = r.f1(1:skipf:end); %f1 = smooth(f1,200);
f2 = r.f2(1:skipf:end); %f2 = smooth(f2,200);
ug = r.ug(1:skipf:end);


yu = [0 .15 .15 0];
x = [0 0 .15 .15];
xb = [.35 .35 .55 .55];
yb = [-.15 .15 .15 -.15];
yl = [0 -.15 -.15 0];


for i=2:length(Xu)
    
    
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
   
%    if(Xl(i) <= 0 )
%        xtmp = [Xl(i) xb(1)/3 xb(1)/3 -xb(1)/3 -xb(1)/3 -Xl(i)];
%        ytmp = [yl(2) yl(2)-.2 yl(2)-.5 yl(2)-.5 yl(2)-.2 yl(2)];
%        fill(xtmp,ytmp,'r');
%    elseif(Xl(i) > 0 && Xu(i) <=0)
%        xtmp = [Xl(i) Xl(i) xb(1)/3 xb(1)/3 -xb(1)/3 -xb(1)/3 -Xl(i) -Xl(i)];
%        ytmp = [0 yl(2) yl(2)-.2 yl(2)-.5 yl(2)-.5 yl(2)-.2 yl(2) 0];
%        fill(xtmp,ytmp,'r');
%    elseif(Xl(i) > 0 && Xu(i) > 0)
%        xtmp = [xb(1)/3   xb(1)/3  Xu(i)  Xu(i) Xl(i) Xl(i)  xb(1)/3   xb(1)/3   -xb(1)/3    -xb(1)/3 -Xl(i) -Xl(i) -Xu(i) -Xu(i)  -xb(1)/3 -xb(1)/3  ];
%        ytmp = [yu(2)+.5  yu(2)+.1 yu(2)   0     0     yl(2) yl(2)-.2  yl(2)-.5   yl(2)-.5   yl(2)-.2  yl(2)  0        0     yu(2)   yu(2)+.1  yu(2)+.5];
%        fill(xtmp,ytmp,'r');
%        
%    end
   
   

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
   
h=text(-.05,-.4, 'Trachea');
set(h,'FontSize',14,'FontWeight','bold');
h = text(-.05,.4, 'Vocal Tract');
set(h,'FontSize',14,'FontWeight','bold');

h = text(-0.05,0.3,['U_g = ' num2str(  round(ug(i))  )]);
set(h,'FontSize',14,'FontWeight','bold');

%h =text(-0.05,0.085,['a_2 = ' num2str(  round(1000*a2(i))  /1000      )]);
%set(h,'FontSize',14,'FontWeight','bold');

%h = text(-0.05,-0.085,['a_1 = ' num2str(  round(1000*a1(i))  /1000      )]);
%set(h,'FontSize',14,'FontWeight','bold');

h = text(-0.05,0.2,['P_i = ' num2str(round(Pi(i)))]);
if(Pi(i)>= 0)
    set(h,'Color',[1 0 0],'FontSize',14,'FontWeight','bold');
else
    set(h,'Color',[0 0 1],'FontSize',14,'FontWeight','bold');
end


h = text(-0.05,-0.25,['P_s = ' num2str(round(Ps(i)))]);
if(Ps(i)>= 0)
    set(h,'Color',[1 0 0],'FontSize',14,'FontWeight','bold');
else
    set(h,'Color',[0 0 1],'FontSize',14,'FontWeight','bold');
end

%tmp = (f1(i)/a1(i) + f2(i)/a2(i))/2; 
%tmp = f2(i)/a2(i);
% tmp =  (1-a2(i)/a1(i))*(Ps(i)-Pi(i)) + Pi(i);
% h = text(-0.05,0.0,['P_g = ' num2str(  round( tmp  ) )]);
% if(tmp>= 0)
%     set(h,'Color',[1 0 0],'FontSize',16,'FontWeight','bold');
% else
%     set(h,'Color',[0 0 1],'FontSize',16,'FontWeight','bold');
% end

tmp2 = f2(i)/a2(i); 
tmp1 = f1(i)/a1(i);
h =text(-0.05,0.087,['P_2 = ' num2str(  round(1*tmp2)       )]);
set(h,'FontSize',14,'FontWeight','bold');

h = text(-0.05,-0.082,['P_1 = ' num2str(  round(1*tmp1)       )]);
set(h,'FontSize',14,'FontWeight','bold');

pgtmp = (tmp1+tmp2)/2;
h = text(-0.05,0.0,['P_g = ' num2str(  round(pgtmp)   )]);
set(h,'Color',[1 0 0],'FontSize',16,'FontWeight','bold');
 if(pgtmp>= 0)
     set(h,'Color',[1 0 0],'FontSize',16,'FontWeight','bold');
 else
     set(h,'Color',[0 0 1],'FontSize',16,'FontWeight','bold');
 end


tmp = (1-a2(i)/a1(i));
h = text(0.4,0.2,['(1- a_2/a_1) = ' num2str( round(1000*tmp)/1000)]);
if(tmp>= 0)
    set(h,'Color',[1 0 0],'FontSize',16,'FontWeight','bold');
else
    set(h,'Color',[0 0 1],'FontSize',16,'FontWeight','bold');
end



  axis([-.8 .8 -.6 .6]);
  set(gca,'Visible','off');
  M(:,i) = getframe;
  hold
end;
