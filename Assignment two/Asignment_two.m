clc; clear all; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Train_data = load('usps-4-9-train.csv'); % load training data
Test_data = load('usps-4-9-test.csv'); % load testing data

[train_m, train_n] = size(Train_data); % size of training data
[test_m, test_n] = size(Test_data); % size of testing data

% Obtain all features of training data
X_train = [ones(train_m, 1) Train_data(:, 1:256)];
Y_train = Train_data(:, 257);

% Obtain all features of testing data
X_test = [ones(test_m, 1) Test_data(:, 1:256)]; 
Y_test = Test_data(:, 257); 

%% Problem One
learning_rate = [10e-8 10e-9 10e-10];
num_lr = length(learning_rate); % obtain the number of learning rate 

% obtain the norm of each w and loss with different learning rate
w_norm = zeros(length(learning_rate), 1); 
loss = w_norm;

% initial optimal weight
initial_w = zeros(size(X_train, 2), 1);

for r = 1:num_lr
    w = Batgrad(X_train, Y_train, 100, initial_w, learning_rate(r), 0);
    w_norm(r) = sqrt(w'*w);
    loss(r) = LossFunc(X_train, Y_train, w);
end

figure
plot(w_norm, loss, 'o-');
title('Batch Gradient Decent with Different Learning Rate')
xlabel('||w||')
ylabel('Loss')
hold off

%% Problem Two
% Compute training accuracy
pre_train = logclassify(sigmoid(X_train*w)); % Prediction on training data
train_error = sum(abs(Y_train - pre_train));
train_accuracy = train_error/train_m;

% Compute testing accuracy
pre_test = logclassify(sigmoid(X_test*w)); % Prediction on testing data
test_error = sum(abs(Y_test - pre_test));
test_accuracy = test_error/test_m;

%% Problem Three

%% Problem Four
