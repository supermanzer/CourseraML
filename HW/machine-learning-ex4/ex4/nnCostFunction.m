function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

K = num_labels; % number of labels as well as output layers
Y = eye(K)(y,:); % rolling up our Y into a 5000x10 matrix

% ---- Part 1 Feed forward propagation ----
a1=[ones(m,1) X]; % input layer - with bias unit
z2 = a1*Theta1'; % layer 2 - z values
a2 = sigmoid(z2);% layer 2 - g(z) or a values
a2 = [ones(size(a2,1),1) a2];  % adding bias unit
z3 = a2*Theta2'; % getting output layer z values
h = sigmoid(z3); % I like keeping this convention since all my cost functions 
                 % use the h variable.
                 
% Computing the costs for all the output layers.  THis should return a 1x10
% vector of costs.
cost = sum((-1*Y).*log(h) - (1-Y).*log(1-h));

% Now we sum across the output layers (K) and divide by 1/m
J = (1/m)*sum(cost);
% Regularizing our Feedforward portion
% Removing bias weights since we do not regularize our bias units.
Theta1_NoBias = Theta1(:,2:end);
Theta2_NoBias = Theta2(:,2:end);

t1Sum = sum(Theta1_NoBias(:).^2);
t2Sum = sum(Theta2_NoBias(:).^2);

reg_term = (lambda/(2*m)) * (t1Sum + t2Sum);

J = J + (lambda/(2*m)) * (t1Sum + t2Sum);
% --------------------------------------------------------

% ---- Part 2 Backpropagation ----
% Following the looped representation of the back propagation algorithm outlined
% int he Backpropagation Algorithm video in week 5.
% Training set already loaded, so first we set our capital delta vales to 0.
Delta1 = 0;
Delta2 = 0;

%============================================================
% Attempting Fully Vectorized version of the below code
%============================================================
%------- Feedforward ------------
bias = ones(m,1);
a1=[bias X]; % Adding bias units
z2 = Theta1*a1';
a2 = [bias sigmoid(z2)'];
z3 = Theta2 * a2';
a3 = sigmoid(z3);

%------- Backprop------------
delta3 = a3' - Y;
delta2 = (Theta2_NoBias' * delta3').*sigmoidGradient(z2);
%delta2 = delta2(:,2:end);
Delta2 = sum(delta3'*a2);
Delta1 = sum(delta2*a1);
% Now we enter our for loop
%for i = 1:m
  % Setting input layer 
%  a1 = [1; X(i,:)']; % Adding bias unit and setting a1 to our input

  % Forward propagation 
 % z2 = Theta1 * a1;
 % a2 = [1; sigmoid(z2)]; % Adding bias unit
 % z3 = Theta2 * a2;
 % a3 = sigmoid(z3); % Here we have our output layer
  
  % Back propagation - 
 % delta3 = a3 - Y(i,:)';
  % We don't check the error associated with our bias units
  % The careful observer will note that we are still including a 
  % bias unit since z2 was calculated from a1, which includes a bias
  % unit.  We'll remove this at the end.
 % delta2 = (Theta2_NoBias' * delta3).*sigmoidGradient(z2);
 % delta2 = delta2(2:end); % Removing the bias unit associated value
  
  % Aggregate our delta values
 % Delta2 += (delta3*a2');
 % Delta1 += (delta2*a1');
%endfor
% Final step to get derivatives, outside of for loop
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% ---- Part 3 Regularization
% We need to avoid including our bias units
Theta1_grad(:,2:end) += ((lambda/m)*Theta1_NoBias);
Theta2_grad(:,2:end) += ((lambda/m)*Theta2_NoBias);






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
