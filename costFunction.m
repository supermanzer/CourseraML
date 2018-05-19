## Copyright (C) 2018 ryan
## Author: ryan <ryan@LinuxHAL>
## Created: 2018-04-29

## Function skeleton to highlight how a cost function should be
## constructed to support powerful optimization
function [jVal, gradient] = costFunction (theta)
  jVal = [code to compute J(theta)];
  
  gradient(1) = [code to compute partial derivative of J(theta) for theta(1)];
  ...
  gradient(n) = [code for ddTheta_n J(theta)];
endfunction
