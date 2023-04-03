function [An] = updateAn_nmf(data_set,core,An,mat_md)
% update one mode factor matrix
    % data_set: concatenate data samples along last mode
    % core:
    % An:
    % mat_md:   matricization mode
    epsilon = 1e-10;
    XnA = getXnA(data_set,An,mat_md);
    Kn_UP = tenmat(XnA,mat_md) * tenmat(core,mat_md)';
    GnA = getGnA(core,An,mat_md);
    Kn_DN = An{mat_md} * tenmat(core,mat_md) * tenmat(GnA,mat_md )';
    Kn = Kn_UP.data ./ (Kn_DN.data+epsilon);
    An{mat_md} = An{mat_md} .* Kn;
    
end

%% ----------------------------------------------------------------
function XnA = getXnA(data_set,An,mat_md)
    order = size(An);order = order(2);
    XnA = data_set;
    for i = 1:order
        if i == mat_md
            continue
        end
        XnA = ttm(XnA,An{i}',[i]);
    end
end

function GnA = getGnA(core,An,mat_md)
    order = size(An);order = order(2);
    GnA = core;
    for i = 1:order
        if i == mat_md
            continue
        end
        GnA = ttm(GnA,An{i}'* An{i},[i]);
    end      
end