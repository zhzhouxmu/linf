function [rank, nFactor] = componentSelection(t,mode)
%factorsInit Initialization for Basis Factors
%   According Algorithm 5
%   t:  tensor of K training samples,I_1 * I_2 * ... * I_N * K 
%   A:  array of N factors I_n * J_n


t_n = tenmat(t,[mode]);
[A_n,Lamda] = eig(double(t_n*t_n'));
[lamda,zeta] = sort(diag(Lamda),'descend');
new_zeta = getJn(lamda,zeta);
rank = length(new_zeta);
nFactor = A_n(:,new_zeta);
nFactor = max(0,nFactor);
end

% -----------------------------------------------
% -----------------------------------------------
function [zeta] = getJn(lamda,zeta)

num = length(lamda);
sums = sum(lamda);
nows = 0;
Jn = 0;
for i = 1:num-1
    nows = nows+lamda(i);
    rate = nows/sums;
    if (rate > 0.98) && (i < num)
      Jn = i;
      break;
    end
end
if Jn == 0
    Jn = num;
end
zeta = zeta(1:Jn);
 
end