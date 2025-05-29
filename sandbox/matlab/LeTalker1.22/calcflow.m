function ug = calcflow(ga,f,b,ar,c,rhoc)
% Flow calculation based on Titze 2002 and 1984
% Author: Brad Story
% Pos = forward pressure wave at subglottis */
% Prg = reflected pressure wave at supraglottis */

	
   Pos = f;
   Prg = b;
   if(length(ar)>44)
    Atrch = ar(end);
   else
       %Atrch = .5;
       Atrch = 1000000;
   end
  
   
   
   Aphrx = ar(1);
   ke = 2*ga/Aphrx*(1-ga/Aphrx);
   kt = 1-ke;
   rho = rhoc/c;
   
   
	Astar = (Atrch * Aphrx) / (Atrch + Aphrx);
	Q = 4.0*kt * (Pos - Prg) / (c*rhoc);
	R = (ga / Astar );
	
	
    %----Titze formulation-----
	if( Q >= 0.0 )	
      ug =  (ga*c/kt) * (-R + sqrt( R*R + Q ));	
	elseif( Q < 0.0 )
      ug =  -(ga*c/kt) * (-R + sqrt( R*R - Q )); 
   end;
   
   
