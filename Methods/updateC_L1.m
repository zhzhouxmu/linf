function [ new_core ] = updateC_L1( data_set,old_core,An )
% update core_tensor using L1 model, same as L2 Model
    % data_set: concatenate data samples along last mode
    % old_core:
    % An
    


%% 

% get the order of sample data
order = size(An);
order = order(2);

X_mdp_A = data_set;
for i = 1:order
    X_mdp_A = ttm(X_mdp_A,An{i}',[i]);
end
G_mdp_A = old_core;
for i = 1:order
    G_mdp_A = ttm(G_mdp_A,(An{i}')*(An{i}),[i]);
end

new_core = old_core .* X_mdp_A ./ G_mdp_A;

end

