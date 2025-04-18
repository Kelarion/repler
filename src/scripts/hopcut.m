function [s, ls] = hopcut(A, b, s0, alpha, T0, period)
%   [s, ls] = hopcut(A, b, s0)
%   
%   Solve QUBO with noisy greedy search
    
    dim = size(A,1);

    % Convert to a nicer canonical form
    thisA = A - diag(diag(A));
    thisb = b + diag(A);
    
    s = s0;
    max_iter = period*(1+floor(log(1e-4/T0)/(log(alpha) + 1e-6)));
    ls = zeros(max_iter,1);
    for i = 1:max_iter
        T = T0*alpha^(floor((i-1)/period));
        for j = 1:dim
            c = thisA(j,:)*s + thisb(j) + 0.5;
            p = 1./(1 + exp(c/T));
            s(j) = 1*(rand() < p);
        end
        ls(i) = s'*A*s + b'*s;
    end

end