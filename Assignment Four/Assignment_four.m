clc; close all; clear all;

%% Load Data
X = load('data.txt');

%% K-means Problem One
K = 5;
[group, sse]= Kmeans(X, K);
iter = length(sse);

figure
plot(1:iter, sse, '-o')
xlabel('iteration')
ylabel('sse')
title(['K-Means Clusering with k = ', num2str(K)])

%% Read Image
for k = 1:K
    figure
    image = ReadImage(group, X, k);
end

%% K-means Problem Two
K2 = 2:10;
L = length(K2);
iteration = 10;
SSE = zeros(L, 1);

for k = 1:L
    minerr = zeros(iteration, 1);
    for itr = 1:iteration
        [G, S] = Kmeans(X, K2(k));
        minerr(itr) = S(end);
    end
    SSE(k) = min(minerr);
end
