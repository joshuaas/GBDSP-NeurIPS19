function [nll,g,H] = SquaredError( w,X,Y )
%SQUAREDERROR Summary of this function goes here
%   Detailed explanation goes here

delta = Y - X * w;
nll = 0.5 * sum(delta(:).^2) ;

if nargout > 1 
  g = X' *X * w - X'* Y ;
end

if nargout > 2
   H = X' * X ;
end

end

