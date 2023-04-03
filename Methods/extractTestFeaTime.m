function [C, A, Out, timeRecord] = extractTestFeaTime(testTensor,model, opts)
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

timeRecord = [];
N = ndims(testTensor)-1;
Nway = size(testTensor); 
sampleNway = Nway(1:N);train_num = Nway(end);
coreNway = zeros(1,N+1);
coreNway(N+1) = Nway(end);
Mnrm = norm(testTensor)/train_num;
% Mnrm = norm(testTensor);

%% ================================================================
% 参数初始化
A = cell(1,N);
if isfield(opts,'A')
    A = opts.A;
else
    error("No Factor Matrix Input!");
end
for n = 1:N
    coreNway(n) = size(A{n},2);
end
if isfield(opts,'C')        
    C = opts.C;       
else 
    if model == "Linf"
        % Wotao Yin's method
%         C = tensor(max(0,randn(coreNway)));
%         C = tensor(C/norm(C)*Mnrm^(1/(N+1)));
        
        % random initialization
        C = tensor(rand(coreNway));
    else
        % random initialization
        C = tensor(rand(coreNway));
    end
end

A0  = A;
C0 = C; Cm = C0;
w = 0; t0 = 1; t = t0;
L = 1; L0 = L;
Asq = cell(1,N);
for n = 1:N
    Asq{n} = A0{n}'*A0{n};
end
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
    tic
    % ----- update Core tensor ------
    if model == "Linf"
       
%             L0 = L;
%             L = 1;
%             for i = 1:N 
%                 L = L * norm(Asq{i}); 
%             end
%             Bsq = Cm; GradC = testTensor;
%             for i = 1:N
%                 Bsq = ttm(Bsq,Asq{i},i);
%                 GradC = ttm(GradC,A{i}',i);
%             end
%             %compute the gradient
%             GradC = Bsq-GradC;
%             C = tensor(max(0,Cm.data-GradC.data/L ));
         
            C = updateC_L2(testTensor,C,A);
         
    elseif model == "L2"
        C = updateC_L2(testTensor,C,A);
    elseif model == "L1"
        C = updateC_L1(testTensor,C,A);
    end
%{
    % ----- update Factor Matrix ------
%     for mat_md = 1:N
%         if model == "Linf"
%             A = updateAn_Linf(testTensor, C, A, mat_md);
%         elseif model == "L1"
%             A = updateAn_L1(testTensor, C, A, mat_md);
%         elseif model == "L2"
%             A = updateAn_L2(testTensor, C, A, mat_md);
%         end
%     end
%}    
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
    
    
    % ----- correction and extrapolation -----
    
    if model == "Linf--"
        t = (1+sqrt(1+4*t0^2))/2;
        for sample_num = 1:Nway(end)
            if obj(sample_num) >= obj0(sample_num)
                % restore C to make the objective nonincreasing
                Cm(:,:,sample_num) = C0(:,:,sample_num);
                C = tensor(max(0,Cm.data-GradC.data/L ));
            else
                %get new extrapolated points
                w = (t0-1)/t; % extrapolation weight
                % choose smaller weight for convergence
                wC = min([w,rw*sqrt(L0)/L]);
                Cm = tensor(C.data+wC*(C.data-C0.data));
                
                C0 = C; t0 = t; obj0 = obj; 
            end
        end %for sample_num = 1:Nway(end)
    end
    
    timeRecord = [timeRecord, toc];
    
end %for k = 1:maxit


end



function [] = printLines()
fprintf("================================================================================\n");
fprintf("  Iter  | RelErr | obj_L2 |  obj_L1 | oj_Linf | OjChg_L2 | OjChg_L1 | OjCg_Linf|\n");
fprintf("================================================================================\n");
end