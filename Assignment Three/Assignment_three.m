clc; close all; clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Train_data = load('knn_train.csv');
Test_data = load('knn_test.csv');

[train_sam, train_fea] = size(Train_data);
[test_sam, test_fea] = size(Test_data);

%% Normalizing Data
XTr = Train_data(:, 2:end); 
Train_mean = mean(XTr);
Train_std = std(XTr);
normal_XTr = (XTr - Train_mean)./Train_std;
X_train = normal_XTr;
Y_train = Train_data(:, 1);

XTe = Test_data(:, 2:end); 
Test_mean = mean(XTe);
Test_std = std(XTe);
normal_XTe = (XTe - Test_mean)./Test_std;
X_test = normal_XTe;
Y_test = Test_data(:, 1);

%% KNN
k = 1:2:60;
numk = length(k); % number of k
Train_error = zeros(numk, 1);
Test_error = Train_error;
delta = Train_error;

for n = 1:numk
    Train_predict = KNNsearch(X_train, X_train, Y_train, k(n));
    Train_error(n) = sum(Train_predict ~= Y_train)/train_sam;
    % leave-one-out error
    for s = 1:train_sam
        leave_one_X = X_train;
        leave_one_Y = Y_train;
        leave_one_X(s,:) = [];
        leave_one_Y(s) = [];
        leave_one_predict = KNNsearch(leave_one_X, X_train(s,:),...
          leave_one_Y, k(n));
        delta(n) = delta(n) + (leave_one_predict ~= Y_train(s));
    end
    delta(n) = delta(n)/train_sam;
    Test_predict = KNNsearch(X_train, X_test, Y_train, k(n));
    Test_error(n) = sum(Test_predict ~= Y_test)/test_sam;
end

figure
hold on
plot(k, Train_error, 'bo-')
plot(k, delta, 'go-')
plot(k, Test_error, 'ro-')
legend('Training Error', 'Leave One Out Error', 'Testing Error')
ylabel('Error Percentage')
xlabel('k')
hold off

%% Decision Tree

