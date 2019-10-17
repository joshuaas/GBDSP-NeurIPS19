function [loss, grad] = loss_dual(x,C,W0,a,b, gamma)
         [n,~] =size(W0) ;
         
         f = x(1 : n) ;
         g = x((n + 1): end) ;
         f_p_g = Oplus(f,g) ;
         Delta = max(f_p_g - C  + gamma * W0,0) ;
	     loss =  0.5 * sumsqr(Delta)/ gamma - sum(a.* f) - sum(b .* g) ;
	     if nargout > 1
            grad_f  =  sum(Delta, 2) /gamma - a ;
            grad_g  = sum(Delta,  1)' /gamma - b ;
            grad    = [grad_f; grad_g] ;
         end
end