%pushbutton1_Callback.m
%Runs the simulation based on current parameter set
%B. Story

h1o = uicontrol('Style','Text','String','Computing...','Position',[320 10 130 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','Type file name','FontWeight','bold','Position',[320 60 100 15],'HorizontalAlignment','Left');
hAudFileName = uicontrol('Style','Edit','Position',[320 80 100 25],'HorizontalAlignment','Left','String',q.AudFileName,'Callback','AssignParams');
   
[p,r]=LeTalker(p,c,N,Fs,ctvect,tavect,plvect);
audn = 0.8*r.po/max(abs(r.po));
h1o = uicontrol('Style','Text','String','Simulation complete','Position',[320 10 130 15],'HorizontalAlignment','Left');
%soundsc(r.po,44100);
PlotLeTalkerWaveforms(r,p,Fs,hwaves,htract);


