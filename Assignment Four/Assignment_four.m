clc; close all; clear all;

%% Load Data
X = load('data.txt');

%% K-means
K = 2;
sse = Kmeans(X, K);
iter = length(sse);

figure
plot(1:iter, sse, '-o')
xlabel('iteration')
ylabel('sse')
title(['K-Means Clusering with k = ', num2str(K)])