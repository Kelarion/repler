function [s, ls] = imf(A, b, s0)
%   [s, ls] = imf(A, b, s0)
%
%   Iterative Maxmimum Flow for solving quadratic unconstrained
%   binary optimization (QUBO) problems:
%       min s'*A*s + b'*s 
%   via majorization-minimization (Konar and Sidiropoulos (2019 ICASSP).
%
%   Provide s0 as an initial guess, optional arguments are a huge pain
%   in this language so sorry about that.

    dim = size(A,1);

    % Convert to a nicer canonical form
    thisA = A - diag(diag(A));
    thisb = b + diag(A);
    
    % Separate positive and negative components
    Am = thisA.*(thisA < 0);
    Ap = -thisA.*(thisA > 0);
    
    s = s0;
    ls = zeros(10,1);
    for i=1:20

        % Compute submodular lower bound
        deez = find(s);
        doze = find(1-s);
        
        active = randperm(sum(s));
        inactive = randperm(sum(1-s));
        
        sig = [deez(active); doze(inactive)];
        [~,gis] = sort(sig);
        
        Aps = Ap(sig, sig);
        
        v = diag(Aps) + sum(triu(Aps,1),1)' + sum(tril(Aps,-1),2);
        v = v(gis);
        
        % Minimize lower bound as a graph cut
        Q = Am;
        p = thisb - v;
        
        ws = p.*(p>0);
        wt = -p.*(p<0) - sum(Q,1)';
        o = zeros(dim,1);
        
        W = [0, ws', 0; o, -Q, wt; 0, o', 0];
        
        G = digraph(W);
        
        [mf, ~, cs, ct] = maxflow(G,1,dim+2);
        idx = ct(ct ~= dim+2) - 1;
        
        s = zeros(dim,1);
        s(idx) = 1;
        % disp([mf, sum(W(cs,ct), "all")])
        if abs(mf - sum(W(cs,ct),"all")) > 1e-6
            disp([mf, sum(W(cs,ct),"all")])
            disp(cs)
            throw(MException)
        end

        % track loss
        ls(i) = s'*A*s + b'*s;

    end
end

