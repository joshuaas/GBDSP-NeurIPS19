function [W,f,g] = Reg_OT(C, W0,gamma)
mfL.Method ='lbfgs' ;
mfL.verbose = false;
mfL.MaxIter = 500 ;
mfL.Display = 'off';
[n,m] = size(C);
f = (1 / n) *  ones(n, 1) ;
g = (1 / m) *  ones(m, 1) ;

a = ones(n, 1) / n ;
b = ones(m, 1) / m ;

L_fun = @(w)loss_dual(w, C, W0, a, b, gamma) ;
sol = minFunc(L_fun, [f ; g], mfL);

f = sol(1 : n) ;
g = sol( (n + 1) : end ) ;


W  =    max( ( Oplus(f, g) - C ) /  gamma + W0, 0 ) ;



end
