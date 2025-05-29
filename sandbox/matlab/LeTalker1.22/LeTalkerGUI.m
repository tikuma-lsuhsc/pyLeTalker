%LeTalkerGUI
%Set q params via GUI
%Author: Brad Story
%10.25.11
%----------------------GUI------------------------------------
hctrl = [];
hwaves = [];
htract = [];

scrsz = get(0,'ScreenSize');
%[left, bottom, width, height]

%for big screen
hctrl = figure('Name','Control Window','Position',[1 scrsz(4)/2 scrsz(3)/5 scrsz(4)/3]);
hwaves = figure('Name','Waveform Window','Position',[scrsz(3)/4 scrsz(4)/6.8 scrsz(3)/3 scrsz(4)/1.5]);
htract = figure('Name','Vocal tract/Trachea Window','Position',[1 scrsz(4)/6 scrsz(3)/5 scrsz(4)/4]);

%for laptop
% hctrl = figure('Name','Control Window','Position',[1 scrsz(4)/1.6 scrsz(3)/3 scrsz(4)/1.6]);
% hwaves = figure('Name','Waveform Window','Position',[scrsz(3)/2.5 scrsz(4)/4 scrsz(3)/3 scrsz(4)/1.5]);
% htract = figure('Name','Vocal tract/Trachea Window','Position',[1 scrsz(4)/12 scrsz(3)/2.5 scrsz(4)/4]);

InitializeLeTalker
load bs_origvowels.mat areas

q.x02 = 0.01;
q.x01 = 0.011;
q.td = 0.05;
q.PL1 = p.PL;
q.PL2 = p.PL;
q.act1 = 0.2;
q.act2 = 0.2;
q.ata1 = 0.2;
q.ata2 = 0.2;
q.CText = 0;
q.CTfreq = 5;
q.TAext = 0;
q.TAfreq = 5;
q.PLext = 0;
q.PLfreq = 5;
q.vow = 6;
q.AudFileName = [];
q.Noise = 0;
audn = [];


figure(hctrl);
clf

%h1o = uicontrol('Style','Text','String','Three-mass model control panel','FontWeight','bold','Position',[190 380 300 25],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','LeTalker Control Panel','FontSize',14,'FontWeight','bold','Position',[130 440 300 20],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','Vers. 1.22, B. Story, U. Arizona (2016)','FontWeight','bold','Position',[130 420 300 15],'HorizontalAlignment','Left');



h1o = uicontrol('Style','Text','String','x02','FontWeight','bold','Position',[10 375 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','x01','FontWeight','bold','Position', [10 350 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','aCT i','FontWeight','bold','Position',[10 325 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','aCT f','FontWeight','bold','Position', [10 300 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','aTA i','FontWeight','bold','Position',[10 275 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','aTA f','FontWeight','bold','Position', [10 250 60 15],'HorizontalAlignment','Left');

h1o = uicontrol('Style','Text','String','PL i','FontWeight','bold','Position',[10 225 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','PL f','FontWeight','bold','Position',[10 200 60 15],'HorizontalAlignment','Left');

%for modulating the PL
h1o = uicontrol('Style','Text','String','PL modf','FontWeight','bold','Position',[200 220 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','PL ext ','FontWeight','bold','Position',[200 195 60 15],'HorizontalAlignment','Left');

h1o = uicontrol('Style','Text','String','vow','FontWeight','bold','Position',[10 150 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','Td','FontWeight','bold','Position', [10 125 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','VT atten','FontWeight','bold','Position', [10 75 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','SUB','FontWeight','bold','Position', [10 50 60 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','SUPRA','FontWeight','bold','Position', [10 25 60 15],'HorizontalAlignment','Left');


%for modulating the CT
h1o = uicontrol('Style','Text','String','CT mod freq (act)','FontWeight','bold','Position',[200 325 100 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','CT extent (act)','FontWeight','bold','Position',[200 300 100 15],'HorizontalAlignment','Left');


%for modulating the TA
h1o = uicontrol('Style','Text','String','TA mod freq (ata)','FontWeight','bold','Position',[200 275 100 15],'HorizontalAlignment','Left');
h1o = uicontrol('Style','Text','String','TA extent (ata)','FontWeight','bold','Position',[200 250 100 15],'HorizontalAlignment','Left');

%For turning on noise generator
h1o = uicontrol('Style','Text','String','Flow Noise  (0 or 1)','FontWeight','bold','Position',[200 350 100 15],'HorizontalAlignment','Left');

h1o = uicontrol('Style','Text','String','Type file name','FontWeight','bold','Position',[320 60 100 15],'HorizontalAlignment','Left');



hx02 = uicontrol('Style','Edit','Position',[90 375 60 25],'HorizontalAlignment','Left','String',num2str(q.x02),'Callback','AssignParams');
hx01 = uicontrol('Style','Edit','Position',[90 350 60 25],'HorizontalAlignment','Left','String',num2str(q.x01),'Callback','AssignParams');
hact1 = uicontrol('Style','Edit','Position',[90 325 60 25],'HorizontalAlignment','Left','String',num2str(q.act1),'Callback','AssignParams');
hact2 = uicontrol('Style','Edit','Position',[90 300 60 25],'HorizontalAlignment','Left','String',num2str(q.act2),'Callback','AssignParams');
hata1 = uicontrol('Style','Edit','Position',[90 275 60 25],'HorizontalAlignment','Left','String',num2str(q.ata1),'Callback','AssignParams');
hata2 = uicontrol('Style','Edit','Position',[90 250 60 25],'HorizontalAlignment','Left','String',num2str(q.ata2),'Callback','AssignParams');

hPL1 = uicontrol('Style','Edit','Position',[90 225 60 25],'HorizontalAlignment','Left','String',num2str(q.PL1),'Callback','AssignParams');
hPL2 = uicontrol('Style','Edit','Position',[90 200 60 25],'HorizontalAlignment','Left','String',num2str(q.PL2),'Callback','AssignParams');
hPLfreq = uicontrol('Style','Edit','Position',[320 215 60 25],'HorizontalAlignment','Left','String',num2str(q.PLfreq),'Callback','AssignParams');
hPLext = uicontrol('Style','Edit','Position',[320 190 60 25],'HorizontalAlignment','Left','String',num2str(q.PLext),'Callback','AssignParams');

hvow = uicontrol('Style','Edit','Position',[90 150 60 25],'HorizontalAlignment','Left','String',num2str(q.vow),'Callback','AssignParams');
htd = uicontrol('Style','Edit','Position',[90 125 60 25],'HorizontalAlignment','Left','String',num2str(q.td),'Callback','AssignParams');
hvtatten = uicontrol('Style','Edit','Position',[90 75 60 25],'HorizontalAlignment','Left','String',num2str(p.vtatten),'Callback','AssignParams');
hSUB = uicontrol('Style','Edit','Position',[90 50 60 25],'HorizontalAlignment','Left','String',num2str(p.SUB),'Callback','AssignParams');
hSUPRA = uicontrol('Style','Edit','Position',[90 20 60 25],'HorizontalAlignment','Left','String',num2str(p.SUPRA),'Callback','AssignParams');


hCTfreq = uicontrol('Style','Edit','Position',[320 315 60 25],'HorizontalAlignment','Left','String',num2str(q.CTfreq),'Callback','AssignParams');
hCText = uicontrol('Style','Edit','Position',[320 290 60 25],'HorizontalAlignment','Left','String',num2str(q.CText),'Callback','AssignParams');

hTAfreq = uicontrol('Style','Edit','Position',[320 265 60 25],'HorizontalAlignment','Left','String',num2str(q.TAfreq),'Callback','AssignParams');
hTAext = uicontrol('Style','Edit','Position',[320 240 60 25],'HorizontalAlignment','Left','String',num2str(q.TAext),'Callback','AssignParams');

hNoise = uicontrol('Style','Edit','Position',[320 340 60 25],'HorizontalAlignment','Left','String',num2str(q.Noise),'Callback','AssignParams');

hAudFileName = uicontrol('Style','Edit','Position',[320 80 100 25],'HorizontalAlignment','Left','String',num2str(q.AudFileName),'Callback','AssignParams');


h = uicontrol('Style','pushbutton','Position',[320 150 60 25],'String','RUN','Callback','pushbutton1_Callback');
h = uicontrol('Style','pushbutton','Position',[390 150 60 25],'String','Quit','Callback','pushbutton2_Callback');
h = uicontrol('Style','pushbutton','Position',[320 125 80 25],'String','Play Audio','Callback','pushbutton3_Callback');
h = uicontrol('Style','pushbutton','Position',[320 100 80 25],'String','Save Audio','Callback','pushbutton4_Callback');

AssignParams;


%---------------------------------------------

