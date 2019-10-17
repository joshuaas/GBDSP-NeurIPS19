function [W, ST, L, S,loss] = GBDSP(X_train,Y_train,Maxsteps,k,lambda, gamma, alpha, K, eval_l, type, L_init,fix)
  %% Parameters
          % parameters for the optimization problem
	  	% min_{S,\tilde{S}, L} \sum_t || Y^t -  X^tLs^t||_2^2 + lambda ||L||_F^2 + gamma || S 
	 	 %  - k * \tilde{S}||_F^2 + \alpha<LG, U>
	  
	  %  k: number of latent tasks
	  
	  %  K: number of transfer groups
	  
	  % eval_l:  whether to type iteration info

	  % type : type of learning tasks :
	  %		   - 'O' if for classification
	  %                - 'R' if for regression
	  

	  % L_type: choose a way to initialize K
	  %                  -  pass the L matrix to this parameter if you need to customize a initialization 
	   %                 - 'common': learn a common L for all latent tasks
	  %                  - 'svd'      :  use a low rank decomposition of a task-specific model


 %% Outputs

 	% W: the per-task parameters W =LS
 	% ST: the auxiliary variable S^\dagger
 	% L, S: use the same notation as the main paper
 	%  loss:  records the loss in every iteration.
	  

%% Dependency
 	%  The L1General Lib from  https://github.com/shyamupa/admm-sentence-reg/tree/master/L1General,
 	%   we have included in our code,see readme for more details.
%% Codes Starts here
%% Preprocessing
warning('off')
mfL.Method ='lbfgs' ;
mfL.verbose = false;
mfL.MaxIter = 15 ;
mfL.Display = 'off';
T = numel(X_train);
D = size(X_train{1},2);
W = zeros(D, T);

  % we use squared loss for regression problem while use squared AUC loss 
    if type == 'R'
    XTX =  cellfun(@(x) x'*x, X_train, 'UniformOutput', false) ;
    XTY  = cellfun(@(x,y) x'*y, X_train, Y_train, 'UniformOutput', false) ;
    else
    if min(Y_train{1}) == -1
        Y_train = cellfun(@(x) (x+1)/2, Y_train, 'UniformOutput', false ) ;
    end
    XTX =  cellfun(@(x,y) calXTLX(x,y), X_train, Y_train , 'UniformOutput', false) ;
    XTY  = cellfun(@(x,y) calXTLY(x,y), X_train, Y_train, 'UniformOutput', false) ;
    end



if  (~ ischar(L_init)) && (ismatrix(L_init))
     
     [l1,l2] = size(L_init) ;
     
     if (l1 ~= D) || (l2 ~=k)
      	error('Incompatible size of the initial L')
      end
      L = L_init ;  
elseif ischar(L_init)
        switch L_init
        	case 'common'
				Term1 = zeros(size(XTX{1},2)) ;
				Term2 = zeros(size(XTX{1},2),1) ;

				for t = 1:T
				  Term1 = Term1 + XTX{t} ;
				  Term2 = Term2 + XTY{t} ;
				end
				theta = (Term1 + lambda * eye(D) ) \ (Term2) ;
				L = theta * ones(1,k)/k ;
		case 'svd'
			         for t = 1:T
			             W(:,t) = (XTX{t} + lambda * eye(D) ) \ XTY{t}  ;
			         end
% 			             U = pcafun(W', k) ; 
			             [U,~,~] = svd(W);
			             
			             L = U(:, 1 : k);
%			             L = W * U;
	        case 'ones'
	        			 L = ones(size(X_train{1},2), k);
        	otherwise
        		error('incorrect choice for initializing L')
        end
else
    error('In correct input of L');
end
	
    if eval_l
        loss = zeros(Maxsteps, 1) ;
    end

    S = zeros(k, T);
    ST = zeros(k, T) ;
      
    Cost =  zeros( size(ST) )  ;
    N_t =zeros(T, 1) ;
    for t  = 1:T
        N_t(t)= size( X_train{t}, 1 );
        if type ~='R'
         N_t(t) = 1 ;
        end
    end
    
    
    for step = 1 : Maxsteps

%% Solve S with the closed form solution
        for t =  1 : T 
            Ut  = X_train{t} * L ; 
            if type == 'R'
                UTU = Ut' * Ut ;
                UTY = Ut' * Y_train{t} ;
            else 
                UTU = calXTLX(Ut, Y_train{t}) ;
                UTY = calXTLY(Ut, Y_train{t}) ;
            end
            S(:, t) = (UTU/N_t(t) + gamma * eye(k) ) \ ( UTY/N_t(t) + gamma * T * ST(:, t)) ;

        end
        if(~fix)
%% update L with a closed form , not suitable for large scale problem.
%             B = zeros(D * k, 1);
%             A = eye( D * k ) * lambda;
%             for t = 1 : T
%                 A = A + 1/N_t(t) * kron(S(:,t)*S(:,t)',X_train{t}'*X_train{t});
%                 A = A + 1 / N_t(t) *  kron( S(: ,t ) * S(: , t )', XTX{t} );
% 
%                 b = X_train{t}'*Y_train{t}*S(:,t)';
%                 
%                 b = XTY{t} * S( : , t )';
% 
%                 B = B + 1 / N_t(t) * b(:);
%             end
%             L = reshape(A\B, size(L) );
%%
%% uopdate L with gradient method scalable for large scale problems

            L_fun = @(w)L_Loss(w, X_train, Y_train, XTX, XTY, S, lambda, k, false, type) ;
        %     disp('learning L')
            L = minFunc(L_fun, L(:), mfL);
            L = reshape(L, D, k) ; 
        end
  %% update ST with the dual of the \ell_F regularized Optimal TLransport (OT) problem
%     disp('learning st')
        Cos_reg = T^2 * gamma/alpha ;
        S_ref  = S /  T;
        ST  = Reg_OT(Cost, S_ref, Cos_reg) ;

    %% udpate U and coefficients, Bdel represents the solution $ U = V^KV{^K^\top}$, 
       %Cost represents  the transportation cost matrix $\mathcal{D}$
        WG = constructBiGraph((ST)) ;
        LW = CalLaplacian(WG);    
        [V, Di] = eig(LW);
        Di = diag(Di);
        [~, ind] = sort(Di, 'ascend');    
        V= V(:,ind(1:K)) ;
        Bdel = V * V';
        Bdel =  diag(Bdel) * ones(1, T  + k) -Bdel;
        Bdel1 = Bdel(1:k, (k+1): end) ; 
        Bdel2 = Bdel((k+1): end, 1:k)';
        Cost = Bdel1 + Bdel2 ;
%     Thresh = alpha * ( Bdel1 + Bdel2) + mu;
        
    if eval_l
       loss( step ) = eval_loss;
       fprintf('loss %d : %3.3f\n', step, loss(step)) ;

    end
    end
W = L * S;
%     W = theta * ones(1, T) ;
 function   ll =  eval_loss()
        ll  = 0 ;
        eig_values = eig(LW) ;
        eig_values = sort(eig_values,'ascend') ;
        block_loss = sum(eig_values((1:K))) ;
        ll = ll + 0.5 * lambda * sumsqr(L) + 0.5 * gamma * sumsqr(S - T * ST) ;
        ll = ll + alpha * block_loss ;
        if type == 'R'
           ll =  ll + eval_reg_loss(L, S) ;

        else 
           ll =  ll + eval_auc_loss(L, S) ;
        end

    end

  %% helper functions

    function L =  CalLaplacian(B)
   		L = (diag( B * ones(size(B,2), 1) )) - B;
    end


     function WG = constructBiGraph(G)
     	 WG = zeros(k+T) ;
     	 AG = abs(G) ;
     	 WG(1:k, (k+1) : end) = AG ;
     	 WG((k+1):end, 1:k )  = AG' ; 

     end



%%   Calculating AUC loss : Here we adopt an efficient AUC evaluation method from one of our earlier work, 
      %    see: 'https://aaai.org/ojs/index.php/AAAI/article/view/4510'  

    function res = calXTLX(A, B)
      np  = sum(B == 1) ;
      nn =  sum(B == 0) ;
      Xp  = A' * B / (np ) ;
      Xn = A' * (1-B) / (nn );
      DD  = B /  np  + (1-B) / nn ;
      res = -Xp * Xn' - Xn * Xp';
      DX = bsxfun(@times, DD, A) ;
      res = res + A' * DX ;
    end

    function res = calXTLY(X, Y)
      np  = sum(Y == 1) ;
      nn =  sum(Y == 0) ;
      Xp  = X' * Y / (np) ;
      Xn = X' * (1-Y)  / (nn);
      res = Xp  - Xn ;
    end

    function lu = eval_auc_loss(L, S)
         lu = 0;
          ww = L * S ;
          for un  = 1:T
              np = sum(Y_train{un} == 1) ;
              nn =sum(Y_train{un} == 0 ) ;
              temp =  (Y_train{un} - X_train{un}  * ( ww(:, un)) ) ;
              lp  = temp' * Y_train{un} / np  ;
              ln = temp'  * (1 - Y_train{un}) /nn;
              llp = (temp .*  Y_train{un} /np)' * temp;
              lln = (temp .* (1- Y_train{un})  / nn )' * temp ;
              lu =  lu +  llp + lln - 2 * lp * ln ;
          end
        
        lu  = lu  *  0.5   ; 

    end


%% Calculating regression loss

    function ll = eval_reg_loss(L, S)
       ll = 0 ;
       ww = L * S ; 
       n_t = cellfun(@(x) size(x,1), X_train) ;

       for idx =  1: T
          ll = ll +  0.5 * sumsqr(Y_train{idx} - X_train{idx} * ww(:, idx)) / n_t(idx) ;
        end
    end
end