function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
h = sigmoid(z);
l = log(h);
J_1 = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h));
grad_1 = (1/m)*X'*(h-y);
reg = eye(length(theta));
reg(1,1) = 0;
J = J_1+(lambda/(2*m))*sum(reg*(theta.^2));
grad = grad_1+(lambda/m)*reg*theta;





% =============================================================

end
