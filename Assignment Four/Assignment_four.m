clc; close all; clear all;

%% Load Data
X = load('data-2.txt');

%% K-means Problem One
K = 4;
[group, sse] = Kmeans(X, K);
iter = length(sse);

figure
plot(1:iter, sse, '-o')
xlabel('iteration')
ylabel('sse')
title(['K-Means Clusering with K = ', num2str(K)])

%% Read Image
for k = 1:K
    figure
    image = ReadImage(group, X, k, 30);
end

%% K-means Problem Two
K2 = 2:10;
L = length(K2);
iteration = 10;
SSE = zeros(L, 1);

for k = 1:L
    minerr = zeros(iteration, 1);
    for itr = 1:iteration
        [~, S] = Kmeans(X, K2(k));
        minerr(itr) = S(end);
    end
    SSE(k) = min(minerr);
end

figure
plot(2:10, SSE, '-o')
xlabel('K')
ylabel('SSE')
title('K-Means Clusering with K = 2 to 10')

%% Single Link
[clust, infoM] = SingleL(X);
clust = cell2mat(clust);
figure
dendrogram(clust)

%% Complete Link
% Same 
