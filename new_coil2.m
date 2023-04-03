% Description 
% 
% 
% BY Zhou Zhehao AT 2019-06-17

%% set parameter 
clc;
clear all; 
close all;
diary off;
ts = datestr(now);
now_time = [ts(8:11),ts(4:6),ts(1:2),'_',ts(13:14),'_',ts(16:17),'_',ts(19:20)];
addpath(genpath('./Tools/'));
addpath(genpath('./Methods'));


data_name = 'NEW_coil8_ms_005-010-015-020';
clstNum = 20;
maxit = 200;
% -------------------------------------------------------------------------
diary(now_time);
fprintf('A new start, come on !!! [%s] ',now_time)
%% process data
%
t = tic;
fprintf('\n===========================================================\n');
fprintf('\t\t\t\t\t [ Process Data ] \n');

%{
% switch data_name
%     case 'mnist'
%        load('.\dataset\MNIST\mnist_data.mat')
%     case 'orl'
%         load('dataset\ORL\orl_data.mat')
%     case 'yale'
%         load('.\dataset\Yale+B\yale_data.mat')
%     case 'coil'
%         load('dataset\COIL20\coil_data.mat')
%     case 'yale_white'
%         load('.\dataset\Yale+B\yale_white.mat')
%     case 'coil_white'
%         load('.\dataset\Yale+B\coil_white.mat')
%     case 'coil_w34'
%         load('.\dataset\Yale+B\coil_w34.mat')
%     case 'coil_w13'
%         load('.\dataset\Yale+B\coil_w13.mat')  
%     otherwise
%         error('No such data name!')
% end
%}

load(['.\dataset\\COIL20\',data_name,'.mat'])
if max(max(test_fea))>200
    trainTensor = tensor(train_tensor/255);
    testTensor = tensor(test_tensor/255);
    train_fea = train_fea/255;
    test_fea = test_fea/255;
else
    trainTensor = tensor(train_tensor);
    testTensor = tensor(test_tensor);
    train_fea = train_fea;
    test_fea = test_fea;
end

%{
% 使用较小的测试集做训练集，验证算法代码有没有错
% 2019-07-29
train_tensor_Temp = test_tensor;
test_tensor = train_tensor;
train_tensor = train_tensor_Temp;
trainTensor= tensor(train_tensor/255);
testTensor = tensor(test_tensor/255);
test_fea_Temp = train_fea/255;
train_fea = test_fea/255;
test_fea = test_fea_Temp;
train_label_Temp = test_label;
test_label = train_label;
train_label = train_label_Temp;
%}

ld_time = toc(t);
fprintf('\t\t\t\t\t [ Process Data ] %f\n',ld_time)
disp('***********************************************************');
%}

%% feature extraction 
%
t = tic;
fprintf('\n===========================================================\n');
fprintf('\t\t\t\t\t [ Eeature Extraction ] \n')
% -------------------------------------------------------------------------
% --------- 统一所有算法的初始点 ---------
N = ndims(trainTensor)-1;
Nway = size(trainTensor);
coreNway = zeros(1,N+1);
coreNway(N+1) = Nway(end);
sampleNway = Nway(1:N);
testNway = size(testTensor);
testCoreNway = zeros(1,N+1);
testCoreNway(N+1) = testNway(end);
A = cell(1,N);
for n = 1:N
    [coreNway(n), A{n}] = componentSelection(trainTensor,n);
end
testCoreNway(1:N) = coreNway(1:N);
for n = 1:N
    A{n} = rand(sampleNway(n), coreNway(n));
end
C_train = tensor(rand(coreNway));
C_test = tensor(rand(testCoreNway));
opt.C = C_train;
opt.A = A;
% --------- --------- ---------
opt.maxit = maxit;
fprintf('#####  [ train Linf ]  #####\n');
model = "Linf";
[trainCore_Linf, An_Linf, Out_Linf] = extractTrainFea(trainTensor,model, opt);
fprintf('#####  [ train L2 ]  #####\n');
model = "L2";
[trainCore_L2, An_L2, Out_L2] = extractTrainFea(trainTensor,model, opt);
fprintf('#####  [ train L1 ]  #####\n');
model = "L1";
[trainCore_L1, An_L1, Out_L1] = extractTrainFea(trainTensor,model, opt);
fprintf('-------------------------------------------------------------\n');
opt.C = C_test;
fprintf('#####  [ test Linf ]  #####\n');
model = "Linf";
opt.A = An_Linf;
[ testCore_Linf, An_Linf, testOut_Linf] = extractTestFea(testTensor,model, opt);
fprintf('#####  [ test L2 ]  #####\n');
model = "L2";
opt.A = An_L2;
[ testCore_L2, An_L2, testOut_L2] = extractTestFea(testTensor,model, opt);
fprintf('#####  [ test L1 ]  #####\n');
model = "L1";
opt.A = An_L1;
[ testCore_L1, An_L1, testOut_L1] = extractTestFea(testTensor,model, opt);
% -------------------------------------------------------------------------
fe_time = toc(t);
fprintf('\t\t\t\t\t [ Eeature Extraction ] %f\n',fe_time);
disp('***********************************************************');
%}
%% vectorize feature
%
t = tic;
fprintf('\n===========================================================\n');
fprintf('\t\t\t\t\t [ Vectorize Feature ] \n')

feature_num = 1;
for o = 1:length(An_L2)
    r = size(An_L2{o});
    feature_num = feature_num * r(2);
end
num_train = size(train_tensor,ndims(train_tensor));
num_test = size(test_tensor,ndims(test_tensor));
trainFeatures = rand(num_train,feature_num);
testFeatures = rand(num_test,feature_num);
% -------------------------------------------------------------------------
trainFea_Linf = trainFeatures;
testFea_Linf = testFeatures;
for i = 1:num_train
    trainFea_Linf(i,:) = reshape(trainCore_Linf(:,:,i),[1,feature_num]);
end
for i = 1:num_test
    testFea_Linf(i,:) = reshape(testCore_Linf(:,:,i),[1,feature_num]);
end
% -------------------------------------------------------------------------
trainFea_L2 = trainFeatures;
testFea_L2 = testFeatures;
for i = 1:num_train
    trainFea_L2(i,:) = reshape(trainCore_L2(:,:,i),[1,feature_num]);
end
for i = 1:num_test
    testFea_L2(i,:) = reshape(testCore_L2(:,:,i),[1,feature_num]);
end
% -------------------------------------------------------------------------
trainFea_L1 = trainFeatures;
testFea_L1 = testFeatures;
for i = 1:num_train
    trainFea_L1(i,:) = reshape(trainCore_L1(:,:,i),[1,feature_num]);
end
for i = 1:num_test
    testFea_L1(i,:) = reshape(testCore_L1(:,:,i),[1,feature_num]);
end
vf_time = toc(t);
fprintf('\t\t\t\t\t [ Vectorize Feature ] %f\n',vf_time)
disp('***********************************************************');
%}


%% performance evaluation
%
t = tic;
fprintf('\n===========================================================\n');
fprintf('\t\t\t\t\t [ Evaluate Performance ] \n');

% ----- Linf ------
[acc, ~] = ...
    AutoClf(train_label,trainFea_Linf,test_label,testFea_Linf);
knnACC.Linf = acc.knn;
svmACC.Linf = acc.svm;
[clstACC.Linf,clstNMI.Linf,~] = ...
    MyKMeans(trainFea_Linf,train_label,clstNum);
% ----- L2 ------
[acc, ~] = ...
    AutoClf(train_label,trainFea_L2,test_label,testFea_L2);
knnACC.L2 = acc.knn;
svmACC.L2 = acc.svm;
[clstACC.L2,clstNMI.L2,~] = ...
    MyKMeans(trainFea_L2,train_label,clstNum);
% ----- L1 ------
[acc, ~] = ...
    AutoClf(train_label,trainFea_L1,test_label,testFea_L1);
knnACC.L1 = acc.knn;
svmACC.L1 = acc.svm;
[clstACC.L1, clstNMI.L1,~] = ...
    MyKMeans(trainFea_L1,train_label,clstNum);
% ----- Base ------
[acc, ~] = ...
    AutoClf(train_label,train_fea,test_label,test_fea);
knnACC.Base = acc.knn;
svmACC.Base = acc.svm;
[clstACC.Base, clstNMI.Base,~] = ...
    MyKMeans(train_fea,train_label,clstNum);

pe_time = toc(t);
fprintf('\t\t\t\t\t [ Evaluate Performance ] %f\n',pe_time)
disp('***********************************************************');
%}

%% happy ending
yourFolder = ['./CacheData/',data_name,'/'];
if ~exist(yourFolder, 'dir')
   mkdir(yourFolder);
end
save(['./CacheData/',data_name,'/',now_time])
diary off