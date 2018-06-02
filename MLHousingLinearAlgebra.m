% This script is intended to capture some of the more 
% useful computations in the Linear Algebra section of
% Andrew Ng's Machine Learning Course.

% Matrix Multiplication for efficient alogrithm testing.

% Suppose we have 4 different house sizes in square feet.
HS = [2104; 1416; 1534; 852];

% We also have 3 competing hypotheses for how to compute
% house prices:
%  h1 = -40 + 0.25 * x
%  h2 = 200 + 0.1 * x
%  h3 = -150 + 0.4 * x

% We can efficiently get predictions from all of our hypotheses
% for all our houses using Matrix Multiplication like so.

% We construct a matrix of our parameters

param = [-40, 200, -150; 0.25, 0.1, 0.4];

% And convert our housing sizes into a matrix with a column of ones

HSM = [ones(4,1) HS];

% Then we can simply multiply on to the other 

y = HSM * param; % Be careful here: param * HSM WOULD NOT BE CORRECT!

% The resulting matrix will have rows for each home size and
% columns for each hypothesis.