function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_pos=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s_pos = c_pos;
k = length(c_pos);
err_vals = zeros(k^2,3);
n=1;
for i = 1:k % looping through each possible C value
  for j = 1:k % looping through each possible sigma value
    model= svmTrain(X, y, c_pos(i), @(x1, x2) gaussianKernel(x1, x2, s_pos(j)));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    err_vals(n,:)=[c_pos(i), s_pos(j), err];
    n = n + 1;
  endfor
endfor
[val, ind] = min(err_vals(:,3));

C = err_vals(ind,1);
sigma = err_vals(ind,2);


% =========================================================================

end
