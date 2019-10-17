function res = Oplus(f,g)
    n = length(f);
    m = length(g);
    res =  f * ones(1,m) + ones(n,1) * g' ;
end