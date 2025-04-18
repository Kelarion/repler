%% 
dims = 10:10:100;
draws = 500;
gap = 1.1; % value of desired maximum is gap * largest eigenvalue

err = zeros(draws, size(dims,2), 3);
ham = zeros(draws, size(dims,2), 3);
marg = zeros(draws, size(dims,2), 3);
runtime = zeros(draws, size(dims,2), 3);

j = 1;
for dim = dims
    H = eye(dim)-1/dim;
    for i = 1:draws

        % % smax = randi([0,1], dim, 1);
        % % if sum(smax) < 1
        % %     smax(1) = 1;
        % % end
        % 
        % % I found that having a balanced s* is much more reliable 
        % % otherwise there are other binary vectors that might have 
        % % a larger value than our desired s* ... 
        % smax = zeros(dim,1); 
        % smax(randsample(dim, floor(dim/2))) = 1;
        % % vmax = smax + randn(dim,1)*0.1;
        % vmax = smax;

        % vvt = H*(vmax*vmax')*H;
        % Q = eye(dim) - vvt/sqrt(sum(vvt.^2,"all"));
        
        % K = wishrnd(H*Q*H, dim)./dim;
        K = wishrnd(eye(dim)-1/dim, dim)./dim;
        
        [V,l] = eig(K);
        [~,idx] = sort(diag(l));
        smax = 1*(V(:,idx(end))>0);
        % 
        % K = K + gap*max(diag(l))*vvt/sqrt(sum(sum(vvt.^2)));
        
        tic;
        [wa, ba] = hopcut(-K, zeros(dim,1), randi([0,1], dim, 1), 1, 1e-4, 20);
        runtime(i,j,1) = toc;

        err(i,j,1) = (wa'*K*wa)/(smax'*K*smax);
        ham(i,j,1) = dim - abs(sum((2*wa-1)'*(2*smax-1)));
        marg(i,j,1) = max(diff(ba));

        tic;
        [wa, ba] = hopcut(-K, zeros(dim,1), randi([0,1], dim, 1), 0.9, 1, 2);
        runtime(i,j,2) = toc;

        err(i,j,2) = (wa'*K*wa)/(smax'*K*smax);
        ham(i,j,2) = dim - abs(sum((2*wa-1)'*(2*smax-1)));
        marg(i,j,2) = max(diff(ba));

        tic;
        [wa, ba] = imf(-K, zeros(dim,1), randi([0,1], dim, 1));
        runtime(i,j,3) = toc;

        err(i,j,3) = (wa'*K*wa)/(smax'*K*smax);
        ham(i,j,3) = dim - abs(sum((2*wa-1)'*(2*smax-1)));
        marg(i,j,3) = max(diff(ba));

    end

    j = j + 1;
end

