function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

posInd = find(y==1);
negInd = find(y==0);

plot(X(posInd,1),X(posInd,2),'k+','Color','b','LineWidth',2);
plot(X(negInd, 1), X(negInd, 2), 'ko', 'MarkerFaceColor', 'r', ...
'MarkerSize', 7);


% =========================================================================



hold off;

end
