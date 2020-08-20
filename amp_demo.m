clc;
clear all;
N = 128; % length of vector to be recovered
M = 64; % number of measurement
phi = (1/sqrt(2))*(normrnd(0,1/sqrt(M),M,N) + 1i*normrnd(0,1/sqrt(M),M,N)); % Sensing matrix construction for theroetical bound
x = zeros(N,1); % Initializing sparse vector to be recovered
k = 13; % Sparsity level
uset = randperm(N,k); 
x(uset) = (rand(k,1) + 1i*rand(k,1))*1e1; % Sparse vector initialized
noise = sqrt(1/2)*(normrnd(0,1,M,1) + 1i*normrnd(0,1,M,1)); % zero mean, unit covariance complex noise vector
var = 1e-11;
noise = sqrt(var)*noise;
y = phi*x + noise; % create measurement
niter = 100; % number of iteration
%% Approximate Message Passing for basis selection
xest = amp(y,k,phi,niter);

[abs(xest) abs(x)]
