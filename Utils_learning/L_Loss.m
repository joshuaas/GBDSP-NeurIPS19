function [ loss, grad ] = L_Loss(L, X, Y, XTX, XTY, S, mu, k, factor, type)
%L_LOSS Summary of this function goes here
%   Detailed explanation goes here
 D =size(X{1}, 2) ;
 L = reshape(L, D, k) ;
 A = zeros(size(L)) ;
 B = zeros(size(L)) ;
 loss = 0 ;
 T = numel(Y) ;
 if type == 'R'
 N_t = cellfun(@(x)size(x, 1), X) ;
 else
  N_t = ones(T,1) ;    
 end
 if factor
    f =  2; 
 else
    f = 1 ;
 end
  ww = L * S ;
  for t = 1 : T
   if type == 'R'    
     loss = loss + sumsqr(Y{t} - X{t}*ww(:, t) ) / N_t(t)  ;
   else
      un = t ;
      np = sum(Y{un} == 1) ;
      nn =sum(Y{un} == 0 ) ;
      temp =  (Y{un} - X{un}  * ( ww(:, un)) ) ;
      lp  = temp' * Y{un} / np  ;
      ln = temp'  * (1 - Y{un}) /nn;
      llp = (temp .*  Y{un} /np)' * temp;
      lln = (temp .* (1- Y{un})  / nn )' * temp ;
      loss =  loss +  llp + lln - 2 * lp * ln ;                  
   end
  end 
  loss  = loss +  mu * sumsqr(L) ;
  if type ~= 'R'
    loss = loss * 0.5 ;
  end
  
  if nargout > 1
      for tt = 1:T
         A  = A + (XTX{tt}* (L * S(:, tt)) ) * S(:, tt)' / N_t(tt);
         B =  B + XTY{tt} * S(:, tt)' /N_t(tt) ;
      end
      grad = A + mu * L - B ;
      grad = grad * f;
      grad = grad(:) ;
  end


end

