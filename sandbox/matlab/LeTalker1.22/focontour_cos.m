function fo = focontour_cos(f01,f02,N)

A = (f01-f02)/2;
B = (f01+f02)/2;

fo = A*cos(2*pi*[0:N-1]/(2*N)) + B;

