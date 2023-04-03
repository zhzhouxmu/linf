function [C, A, Out] = extractTestFea(testTensor,model, opts)
%EXTRACTTESTFEA 提取测试集特征
%   testTensor: 测试集集数据张量
%   model:  模型选择：{"L2", "L1", "Linf"}
%   opts:   算法参数
%   ----------------------------------------------
%   C:  核心张量
%   A:  因子矩阵
%   Out: 算法迭代过程信息
%   ----------------------------------------------
%   Zhouzh 2019.07.02
%   ==============================================



%% ================================================================
% 基础参数设置
if isfield(opts,'maxit')  maxit = opts.maxit; else maxit = 500;   end
if isfield(opts,'tol')    tol = opts.tol;     else tol = 0.001;   end
if isfield(opts,'rw')     rw = opts.rw;       else rw = 1;     end

N = ndims(testTensor)-1;
Nway = size(testTensor); 
sampleNway = Nway(1:N);train_num = Nway(end);
coreNway = zeros(1,N+1);
coreNway(N+1) = Nway(end);
Mnrm = norm(testTensor)/train_num;
% Mnrm = norm(trainTensor);
%% ================================================================
% 参数初始化
A = cell(1,N);
for n = 1:N
    [coreNway(n), A{n}] = componentSelection(testTensor,n);
end
if isfield(opts,'C')        
    C = opts.C;       
else
    if model == "Linf"
%         % Wotao Yin's method
%         C = tensor(max(0,randn(coreNway)));
%         C = tensor(C/norm(C)*Mnrm^(1/(N+1)));
        
        % random initialization
        C = tensor(rand(coreNway));
    else
        % random initialization
        C = tensor(rand(coreNway));
    end
end
if isfield(opts,'A')
    A = opts.A;
else
    for n = 1:N
        if model == "Linf"
%             % Wotao Yin's method
%             A{n} = max(0,randn(sampleNway(n),coreNway(n)));
%             A{n} = A{n}/norm(A{n},'fro')*Mnrm^(1/(N+1));
            
            % random initialization
            A{n} = rand(sampleNway(n), coreNway(n));
        else
            % random initialization
            A{n} = rand(sampleNway(n), coreNway(n));
        end
    end
end
A0  = A;
C_k1 = C; Cm = C; C_k2 = C;
w = 0; t0 = 1; t = t0;
L = 1; L0 = L;
% Asq = cell(1,N);
% for n = 1:N
%     Asq{n} = A0{n}'*A0{n};
% end
%% 算法部分
obj0 = 1e10*ones(Nway(end),1);
Out.obj_Linf = zeros(10,1);
Out.obj_L2 = zeros(10,1);
Out.obj_L1 = zeros(10,1);
Out.rel_err = zeros(10,1);
Out.t = zeros(10,1);

printLines();
%% ================================================================
% Iterations update 
% =========================================================================
for k = 1:maxit
    % ----- update Core tensor ------
    if model == "Linf"
        % 使用BCD算法更新
        t = (1+sqrt(1+t0^2))/2;
        wm_k1 = (t0 - 1)/t;
        L = 1;
        for i = 1:N 
            L = L * norm(A{i}'*A{i}); 
        end
        w_k1 = min(wm_k1, rw*sqrt(L0/L));
        if k==1  
            Cm_k1 = C_k1 ;
        else
            Cm_k1 = C_k1 + w_k1*(C_k1 - C_k2);
        end
        GradC = C_k1;temp = testTensor;
        for n = 1:N
            GradC = ttm(GradC, A{n}'*A{n}, n);
            temp = ttm(temp, A{n}', n);
        end
        GradC = GradC - temp;
        C = tensor(max(0, Cm_k1.data - (GradC.data/L)));
              
              % 使用L2的方法来更新
%             C = updateC_L2(trainTensor,C,A);
    elseif model == "L2"
        C = updateC_L2(testTensor,C,A);
    elseif model == "L1"
        C = updateC_L1(testTensor,C,A);
    end
    
    % ----- diagnostics, reporting, stopping checks -----
    ttm_order = linspace(1,N,N);
    errorTensor = ttm(C,A,ttm_order) - testTensor;
    rel_err = norm(errorTensor)/norm(testTensor);
    errList = errorTensor.^2;
    errList = sum(errList.data,ttm_order);
    FNormList = sqrt(errList);
    obj = errList;
    
    % reporting
    Out.obj_Linf(k) = max(FNormList);
    Out.obj_L2(k) = sum(errList);
    Out.obj_L1(k) = sum(FNormList);
    Out.rel_err(k) = rel_err;
    Out.t(k) = t;  
    
    % ----- correction and extrapolation -----
    if model == "Linf"
        for sample_num = 1:Nway(end)
            if obj(sample_num) >= obj0(sample_num)
                % restore C to make the objective nonincreasing
                Cm_k1(:,:,sample_num) = C_k1(:,:,sample_num);
                C = tensor(max(0, Cm_k1.data - (GradC.data/L)));
            end
        end %for sample_num = 1:Nway(end)
         L0 = L;
        C_k2 = C_k1;
        C_k1 = C;
        % ----- diagnostics, reporting, stopping checks -----
        ttm_order = linspace(1,N,N);
        errorTensor = ttm(C,A,ttm_order) - trainTensor;
        rel_err = norm(errorTensor)/norm(trainTensor);
        errList = errorTensor.^2;
        errList = sum(errList.data,ttm_order);
        FNormList = sqrt(errList);
        obj = errList;

        % reporting
        Out.obj_Linf(k) = max(FNormList);
        Out.obj_L2(k) = sum(errList);
        Out.obj_L1(k) = sum(FNormList);
        Out.rel_err(k) = rel_err;
        Out.t(k) = t; 
    end
    
    % check stopping criterion 
    if k > maxit
        break;
    elseif k>1
        objChg_Linf = abs(Out.obj_Linf(k) - Out.obj_Linf(k-1)) / Out.obj_Linf(k-1);
        objChg_L2 = abs(Out.obj_L2(k) - Out.obj_L2(k-1) ) / Out.obj_L2(k-1);
        objChg_L1 = abs(Out.obj_L1(k) - Out.obj_L1(k-1) ) / Out.obj_L1(k-1);
        rel_errChg = abs(Out.rel_err(k) - Out.rel_err(k-1) ) / Out.rel_err(k-1);

        fprintf(" %3d th | %1.3f | %7d | %7d | % 4.3f | %1.6f | %1.6f | %1.6f |\n",...
                k,Out.rel_err(k),...
                round(Out.obj_L2(k)),round(Out.obj_L1(k)),Out.obj_Linf(k),...
                objChg_L2,objChg_L1,objChg_Linf);
        if (objChg_Linf < tol) && (model == "Linf") && k>50
            break;
        elseif (objChg_L2 < tol) && (model == "L2") && k>50
            break;
        elseif (objChg_L1 < tol) && (model == "L1") && k>50
            break;
        end
        if mod(k,20) == 0   printLines;        end
 
    end
    
end %for k = 1:maxit


end



function [] = printLines()
fprintf("================================================================================\n");
fprintf("  Iter  | RelErr | obj_L2 |  obj_L1 | oj_Linf | OjChg_L2 | OjChg_L1 | OjCg_Linf|\n");
fprintf("================================================================================\n");
end