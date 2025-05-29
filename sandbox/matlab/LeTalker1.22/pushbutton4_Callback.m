%pushbutton4_Callback.m
%B. Story
%clf
%clear hx02r hx02l hxibr hxibl hnpr hnpl haepi hlepi hpgap hvow1 hvow2 hdxnew htd hfo h1o h p c

if(isempty(audn) == 0)
    
    %hAudFileName = uicontrol('Style','Edit','Position',[320 60 130 15],'HorizontalAlignment','Left','String',q.AudFileName,'Callback','AssignParams');


    if(isempty(q.AudFileName) == 0)

        q.AudFileName = [q.AudFileName '.wav'];
        wavwrite(audn,44100,16,q.AudFileName);
        h1o = uicontrol('Style','Text','String',[q.AudFileName ' saved'],'FontWeight','bold','Position',[320 60 100 15],'HorizontalAlignment','Left');
        q.AudFileName = [];
    end


else
    h1o = uicontrol('Style','Text','String','Cannot write to file until RUN','Position',[320 10 130 15],'HorizontalAlignment','Left');
end

