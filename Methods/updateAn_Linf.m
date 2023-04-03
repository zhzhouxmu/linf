function [An] = updateAn_Linf(data_set,core,An,mat_md)
% update one mode factor matrix
    % data_set: concatenate data samples along last mode
    % core:
    % An:
    % mat_md:   matricization mode
    % ------------------------------------------------
    % * Modify the 'getKronAList' function
    % * Change cvx precision (low -> default)
    %   (by zhouzh on 2019-06-24)
    % 
    
num_train = size(data_set);num_train = num_train(end);
order = size(An);order = order(2);
size_Ai = size(An{mat_md});

KronA_list = getKronAList(order,mat_md);
KronA = getKronA(An,KronA_list);

GnA = getGnA(core,KronA,mat_md,num_train);
% GnA
mat_X = getMatX(data_set,mat_md,num_train);

element_num = size_Ai(1)*size_Ai(2);

% cvx_solver mosek
% % cvx_precision(0.1)
% cvx_precision low
cvx_quiet(true);
cvx_begin
    variable x(element_num) nonnegative;
    variable t;
    minimize(t);
    subject to
        for i = 1:num_train
            norm(mat_X(:,:,i) - ...\
                reshape(x,[size_Ai(1),size_Ai(2)])* GnA(:,:,i),'fro') <= t;
        end
cvx_end

An{mat_md} = reshape(x,size_Ai(1),size_Ai(2));

end

%% ----------------------------------------------------------------
function [KronA_list] = getKronAList(order,mat_md)
    list  = linspace(1,order,order);
    la_list = list(mat_md+1:end);
    le_list = list(1:mat_md-1);
    left_list = fliplr(la_list);
    right_list = fliplr(le_list);
    KronA_list = [left_list,right_list];
end

function [KronA] = getKronA(An,kronA_list)
    KronA = An{kronA_list(1)};
    for i =2:length(kronA_list)
        KronA = kron(KronA,An{kronA_list(i)});
    end
end

function [GnA] = getGnA(core,KronA,mat_md,num_train)
   dim = ndims(core);
   num_feature = size(core);
   temp_matrix = zeros(num_feature(1:end-1));
   temp_matrix = reshape(temp_matrix,num_feature(mat_md),[]);
   size_tm = size(temp_matrix*KronA');
   GnA = zeros([size_tm,num_train]);
   
   str = ':';
   for i = 1:dim-2
       str = [str,',:'];
   end
   
   for i = 1:num_train
       each_core = eval(['core(',str,',',num2str(i),');']);
       GnA(:,:,i) = tenmat(each_core,[mat_md])*KronA';
   end
end

function [mat_X] = getMatX(data_set,mat_md,num_train)
    dim = ndims(data_set);
    num_feature = size(data_set);
    temp_matrix = zeros(num_feature(1:end-1));
    temp_matrix = reshape(temp_matrix,num_feature(mat_md),[]);
    size_tm = size(temp_matrix);
    mat_X = zeros([size_tm,num_train]);
    
    str = ':';
    for i =1:dim-2
        str = [str,',:'];
    end
    
    for i=1:num_train
        each_sample = eval(['data_set(',str,',',num2str(i),');']);
        mat_X(:,:,i) = tenmat(each_sample,[mat_md]);
    end
end

