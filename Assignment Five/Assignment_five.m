clc; clear all; 

%% Import data
data = dlmread('test-data-for-MDP-1.txt');
% data = [3 2 0;... % states, actions, 0 is irrelevant
%     0.2 0.8 0; 0 0.2 0.8; 1 0 0;... % action one
%     0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9;... % action two
%     -1 -1 0]; % reward for each states

%% Initialization
states = data(1,1); actions = data(1,2); 
% obtain transition matries for all states
M = data(2:end-1,:); 
	
% Transition model/matrix
T = zeros(states, states, actions);
for a = 1:actions
	T(:,:,a) = M((1+(a-1)*states):a*states,:);
end
	
% Reward 
R = data(end,:); 
R = R';
	
%% \gamma = 0.1
gamma1 = 0.1; % discount factor one
[U1, Pi_1, iter1] = ValueIte(R, gamma1, states, actions, T);
	
%% \beta = 0.5
gamma2 = 0.5; % discount factor two
[U2, Pi_2, iter2] = ValueIte(R, gamma2, states, actions, T);
	
%% \beta = 0.9
gamma3 = 0.9; % discount factor three
[U3, Pi_3, iter3] = ValueIte(R, gamma3, states, actions, T);
	
	
%% Value Iteration Algorithm
function [U, Pi, iter] = ValueIte(R, gamma, states, actions, T)
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
	
fprintf(['With gamma = ', num2str(gamma),'\n'])
fprintf('the optimal utility is \n')
display(U')
	
% calculate optimal policy
P = zeros(states, actions);
for a = 1:actions
	P(:,a) = T(:,:,a)*U;
end
[~, Pi] = max(P,[],2);
fprintf('the optimal policy is \n')
display(Pi')
	
end
