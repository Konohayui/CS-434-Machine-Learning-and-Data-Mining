function [clust, infoMatrix] = SingleL(X)
% test example
% X = [1 1; 1.5 1.5; 5 5; 3 4; 4 4; 3 3.5];
% X = [1 2; 2.5 4.5; 2 2;4 1.5; 4 2.5];
% http://people.revoledu.com/kardi/tutorial/Clustering/Online-Hierarchical-Clustering.html

samples = size(X, 1);
clust = []; % initializing cluster
c = samples; % initializing new number cluster
distance = pdist(X); % compute pair wise distance
DistValue = squareform(distance); % comstruct a matrix for each data point
DistMatrix = DistValue;

% comstruct a matrix to store all information
infoMatrix = zeros(samples + 1, samples + 1); 
infoMatrix(1, :) = 0:samples;
infoMatrix(2:end, 1) = 1:samples;
infoMatrix(2:end, 2:end) = DistMatrix;

while true
    [minD, pos]= min(distance); % find the minimum distance
    DistN = DistMatrix; % rename distance matrix
    DistN(triu(true(size(DistN)))) = NaN; % avoid duplicate distance
    % find the index of the min distance
    [min_i, min_j] = find(DistN == minD,1); 
%     fprintf(['Min distance between ',num2str(min_i),' and ',...
%         num2str(min_j), ' is ', num2str(minD),'\n'])
    
    if size(min_i) ~= 0 % to make sure obtain index of min distance 
        c = c + 1; % update cluster
        % append new cluster
        clust = [clust; {infoMatrix(1, min_i + 1), ...
            infoMatrix(1, min_j + 1), minD}]; 
        % update cluster number to infoM
        infoMatrix(1, min_j + 1) = c;
        infoMatrix(min_j + 1, 1) = c;
        % Complete link
        % max_dist = max(DistM(:,min_i), DistM(:,min_j));
        min_dist = min(DistMatrix(:,min_i), DistMatrix(:,min_j));
        min_dist(min_j) = 0;
        % update new distance to cluster
        DistMatrix(:,min_j) = min_dist;
        DistMatrix(min_j,:) = min_dist';
    end
    
    % delete old distance
    DistMatrix(min_i,:) = [];
    DistMatrix(:,min_i) = [];
    infoMatrix(min_i + 1,:) = [];
    infoMatrix(:,min_i + 1) = [];
    distance(pos) = [];
    infoMatrix(2:end, 2:end) = DistMatrix;
%     display(infoMatrix)
    
    [m, n] = size(DistMatrix);
    if m == 2 && n == 2
        break;
    end
    
end

end