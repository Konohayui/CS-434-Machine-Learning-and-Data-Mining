function Prediction = KNNsearch(Xtrain, Xtest, Y, k)
Train_samples = size(Xtrain, 1);
Test_samples = size(Xtest, 1);
Prediction = zeros(Test_samples, 1);

for i = 1:Test_samples
    dist = zeros(Train_samples, 1);
    for j = 1:Train_samples
        dist(j) = norm(Xtest(i, :) - Xtrain(j, :));
    end
    [distance, index] = sort(dist);
    close_dist = distance(1:k);
    close_ind = index(1:k);
    Prediction(i) = mode(Y(close_ind));
end

% Note:
% Each row of dist obtain the distance between
% ith row of testing data and jth row training data.
% Such as, dist(1, 1) is the distance between 
% the first testing sample and the first training sample;
% dist(1, 2) is the distance between
% the first testing sample and the second training sample.

end