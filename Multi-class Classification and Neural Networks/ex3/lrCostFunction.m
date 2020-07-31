function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
z = X * theta;
h = sigmoid(z);
reg = (lambda/(2*m)) * sum(theta(2:end).^2);
J = (1/m)*sum((-y.*log(h))-((1-y).*log(1-h))) + reg;
grad(1) = (1/m) * (X(:,1)'*(h-y));                            
grad(2:end) = (1/m) * (X(:,2:end)'*(h-y)) + (lambda/m)*theta(2:end);
  






% =============================================================

grad = grad(:);

end
