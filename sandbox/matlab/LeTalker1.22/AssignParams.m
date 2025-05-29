%AssignParams
%Author: Brad Story
%10.05.12


q.x02 = str2double(get(hx02,'string'));
q.x01 = str2double(get(hx01,'string'));
q.act1 = str2double(get(hact1,'string'));
q.act2 = str2double(get(hact2,'string'));
q.ata1 = str2double(get(hata1,'string'));
q.ata2 = str2double(get(hata2,'string'));

q.vow = str2num(get(hvow,'string'));

q.td = str2double(get(htd,'string'));
q.PL1 = str2double(get(hPL1,'string'));
q.PL2 = str2double(get(hPL2,'string'));

q.PLfreq = str2double(get(hPLfreq,'string'));
q.PLext = str2double(get(hPLext,'string'));

q.CTfreq = str2double(get(hCTfreq,'string'));
q.CText = str2double(get(hCText,'string'));

q.TAfreq = str2double(get(hTAfreq,'string'));
q.TAext = str2double(get(hTAext,'string'));

p.vtatten = str2double(get(hvtatten,'string'));
p.SUB = str2double(get(hSUB,'string'));
p.SUPRA = str2double(get(hSUPRA,'string'));
p.Noise = str2double(get(hNoise,'string'));

q.AudFileName = get(hAudFileName,'string');



               
Fs=44100;           % sampling frequency
N= round(q.td*Fs);    
tm=[0:1/Fs:(N-1)/Fs];

ctvect = MovementTrajCos([1 N],[q.act1 q.act2]);
ctvect = (q.CText*ctvect').*sin(2*pi*q.CTfreq*tm)+ctvect';

tavect = MovementTrajCos([1 N],[q.ata1 q.ata2]);
tavect = (q.TAext*tavect').*sin(2*pi*q.TAfreq*tm)+tavect';

plvect = MovementTrajCos([1 N],[q.PL1 q.PL2]);
plvect = (q.PLext*plvect').*sin(2*pi*q.PLfreq*tm)+plvect';


p.x02 = q.x02;
p.x01 = q.x01;
p.PL = q.PL1;
if(q.vow > 0 && q.vow < 12)
    p.ar(1:44) = areas(q.vow,:);
    p.ar(1:44) = smooth(p.ar(1:44),3);
else
    p.ar(1:44) = 3*ones(1,44);
end
    


