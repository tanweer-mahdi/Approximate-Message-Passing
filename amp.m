function xest = amp(yn,sparsity,phi,niter)

N = size(phi,2);
M = size(phi,1);
tau = sqrt(2*log10(M/sparsity)); % needs to be tuned in case of unknown prior sparsity level

%% Approximate Message Passing for basis selection
% Initializing
xest = zeros(N,1);
yp = yn;
z = yp;
eta = @(x,beta) (x./abs(x)).*(abs(x)-beta).*(abs(x)-beta > 0); % denoising function
for iter=1:niter
   sigma = norm(z,2)/sqrt(M);
   xest = eta(xest + phi'*z, tau*sigma);
   tmp = sum(abs(xest) > 0);
   z = yp - phi*xest + (tmp/M)*z;
end

%[abs(xest) abs(x)]
end