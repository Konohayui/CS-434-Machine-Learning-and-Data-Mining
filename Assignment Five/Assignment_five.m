clc; clear all; 

%% Import data
data = dlmread('test-data-for-MDP-1.txt');
% data = [3 2 0;... % states, actions, 0 is irrelevant
%     0.2 0.8 0; 0 0.2 0.8; 1 0 0;... % action one
%     0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9;... % action two
%     -1 -1 0]; % reward for each state

%% Initialization
states = data(1,1); actions = data(1,2); 
% obtain transition matries for all states
M = data(2:end-1, :); 

% Transition model/matrix
T = zeros(states, states, actions);
for a = 1:actions
    T(:,:,a) = M((1+(a-1)*states):a*states,:);
end

% Reward 
R = data(end, :); 
R = R';

%% \beta = 0.1
beta1 = 0.1; % discount factor
[U1, iter1] = ValueIte(R, beta1, states, actions, T);
fprintf(['With beta = ', num2str(beta1),'\n'])
fprintf('the optimal utility is \n')
display(U1)

% calculate optimal policy
P1 = zeros(states, actions);
for a = 1:actions
    P1(:,a) = T(:,:,a)*U1;
end
[~, Pi_1] = max(P1,[],2); % optimal policy for \beta = 0.1
fprintf('the optimal policy is \n')
display(Pi_1)

%% \beta = 0.9
beta2 = 0.9; % discount factor
[U2, iter2] = ValueIte(R, beta2, states, actions, T);
fprintf(['With beta = ', num2str(beta2),'\n'])
fprintf('the optimal utility is \n')
display(U2)

% calculate optimal policy
P2 = zeros(states, actions);
for a = 1:actions
    P2(:,a) = T(:,:,a)*U2;
end
[~, Pi_2] = max(P2,[],2); % optimal policy for \beta = 0.9
fprintf('the optimal policy is \n')
display(Pi_2)

%% Value Iteration Algorithm
function [U, iter] = ValueIte(R, gamma, states, actions, T)
converge = 0; % logical statement for convergence
epsilon = 1e-10;
delta = epsilon*(1 - gamma)^2/(2*gamma^2);
iter = 0; % initial iteration
U = R; % initial utility

while ~converge
    iter = iter + 1;
    old_U = U;
    
    % horizon decision
    M = zeros(states,actions);
    for a = 1:actions
        M(:,a) = T(:,:,a)*old_U;
    end
    
    % Bellman Equation
    U = R + gamma*max(M,[],2);
    
    % check convergence
    if abs(sum(U - old_U)) < delta
        converge = 1;
    end
    
end

end
