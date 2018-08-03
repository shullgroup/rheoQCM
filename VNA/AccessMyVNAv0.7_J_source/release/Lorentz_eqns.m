% FITTING FUNCTIONS

% Parameters for fitting a single Lorentz peak
% p=[f(MHz) gamma(1e4 Hz) phi(1e2 deg.) Gmax Goffset Boffset];
%single, 6 parameters

% Parameters for fitting two Lorentz peaks
% p=[f1(MHz) gamma1(1e4 Hz) phi1(1e2 deg.) Gmax1 Goffset 
%    f2(MHz) gamma2(1e4 Hz) phi2(1e2 deg.) Gmax2 Boffset]; 
% double 10par 

% Parameters for fitting three Lorentz peaks
% p=[f1(MHz) gamma1(1e4 Hz) phi1(1e2 deg.) Gmax1 Goffset 
%    f2(MHz) gamma2(1e4 Hz) phi2(1e2 deg.) Gmax2 
%    f3(MHz) gamma3(1e4 Hz) phi3(1e2 deg.) Gmax3 Boffset];
% triple 14par

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Equation to fit a conductance resonance peak
lfun4c=@(p,x)...
    p(4).*((((x.^2).*((2.*p(2).*1e4).^2))./(((((p(1).*1e6).^2)-...
    (x.^2)).^2)+((x.^2).*((2.*p(2).*1e4).^2)))).*cosd(p(3)./1e2)-...
    ((((p(1).*1e6).^2-x.^2)).*x.*(2.*p(2).*1e4))./(((((p(1).*1e6).^2)-...
    (x.^2)).^2)+((x.^2).*((2.*p(2).*1e4).^2))).*sind(p(3)./1e2));

%Equation to fit a susceptance resonance peak
lfun4s=@(p,x)...
    -p(4).*(-(((x.^2).*((2.*p(2).*1e4).^2))./(((((p(1).*1e6).^2)-...
    (x.^2)).^2)+((x.^2).*((2.*p(2).*1e4).^2)))).*sind(p(3)./1e2)-...
    ((((p(1).*1e6).^2-x.^2)).*x.*(2.*p(2).*1e4))./(((((p(1).*1e6).^2)-...
    (x.^2)).^2)+((x.^2).*((2.*p(2).*1e4).^2))).*cosd(p(3)./1e2));    

%Function for fitting a single Lorentz peak
lfun4_both_1=@(p,x) [lfun4c(p(1:4),x)+p(5),lfun4s(p(1:4),x)+p(6)];

%Function for fitting two Lorentz peaks
lfun4c_2=@(p,x) lfun4c(p(1:4),x)+lfun4c(p(6:9),x)+p(5);
lfun4s_2=@(p,x) lfun4s(p(1:4),x)+lfun4s(p(6:9),x)+p(5);    
lfun4_both_2=@(p,x) [lfun4c_2(p(1:9),x),lfun4s_2([p(1:4),p(10),p(6:9)],x)];

%Function for fitting three Lorentz peaks
lfun4c_3=@(p,x) lfun4c(p(1:4),x)+lfun4c(p(6:9),x)+lfun4c(p(10:13),x)+p(5);
lfun4s_3=@(p,x) lfun4s(p(1:4),x)+lfun4s(p(6:9),x)+lfun4s(p(10:13),x)+p(5);    
lfun4_both_3=@(p,x)...
    [lfun4c_3(p(1:13),x),lfun4s_3([p(1:4) p(14) p(6:13)],x)];