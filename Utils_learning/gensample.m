num_T = 100;
num_D = 300;
for i = 1:num_T
   X{i} = rand(1000, num_D) ;
end
%% for regression
for i = 1:num_T
   Y{i} = X{i} * rand(num_D,1);
end
%% for AUC
for i = 1:numel(Y)
   temp  = Y{i} ;
   [~,idx] =sort(Y{i},'ascend') ;
   thre = temp(idx(800)) ;
   temp(temp <= thre) = 0 ;
   temp(temp > thre) = 1 ;
   Y{i} =temp ; 
end 
