  
   function lu = eval_auc(X,Y,L, S)
         T = length(X);
         lu = 0;
          ww = L * S ;
          for un  = 1:T
              np = sum(Y{un} == 1) ;
              nn =sum(Y{un} == 0 ) ;
              temp =  (Y{un} - X{un}  * ( ww(:, un)) ) ;
              lp  = temp' * Y{un} / np  ;
              ln = temp'  * (1 - Y{un}) /nn;
              llp = (temp .*  Y{un} /np)' * temp;
              lln = (temp .* (1- Y{un})  / nn )' * temp ;
              lu =  lu +  llp + lln - 2 * lp * ln ;
          end
        
        lu  = lu  *  0.5   ; 

    end
