function [W,f,g] = Reg_OT(C, W0,gamma)
mfL.Method ='lbfgs' ;
mfL.verbose = false;
mfL.MaxIter = 100;

[n,m] = size(C);
f = zeros(n,1) ;
g = zeros(m,1) ;

a = ones(n,1) / n ;
b = ones(m,1) / m ;

L_fun = @(w)loss_dual(w, C, W0, a,b,gamma) ;
sol = minFunc(L_fun, [f;g], mfL);

f = sol(1:n) ;
g = sol((n + 1) : end) ;


W  =    max((Oplus(f,g) - C) / gamma + W0, 0 ) ;


end
