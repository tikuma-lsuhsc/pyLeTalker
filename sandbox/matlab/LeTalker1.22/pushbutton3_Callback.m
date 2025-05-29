%pushbutton3_Callback.m
%B. Story

%clf
%clear hx02r hx02l hxibr hxibl hnpr hnpl haepi hlepi hpgap hvow1 hvow2 hdxnew htd hfo h1o h p c
if(isempty(r.po) == 0)
    soundsc([zeros(1,25000) r.po zeros(1,25000)],44100);
    h1o = uicontrol('Style','Text','String','Playing output','Position',[320 25 100 15],'HorizontalAlignment','Left');
else
    h1o = uicontrol('Style','Text','String','Cannot play','Position',[320 25 100 15],'HorizontalAlignment','Left');
end
    
    