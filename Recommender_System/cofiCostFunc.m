function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%
%Notes:  X -> num_movies  x num_features matrix of movie features
%        Theta -> num_users  x num_features matrix of user features
%        Y -> num_movies x num_users matrix of user ratings of movies
%        R -> num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

%Calculate Cost Function
J = 0.5 * sum(sum((X * Theta' - Y).^2 .* R));
J = J + lambda / 2 * (sum(sum(X.^2))  + sum(sum(Theta.^2)));

%Calculate Gradients
X_grad = ((X * Theta' - Y) .* R) * Theta;
X_grad = X_grad + lambda .* X;
Theta_grad = ((X * Theta' - Y) .* R)' * X;
Theta_grad = Theta_grad + lambda.*Theta; 

%fmincg function needs all parameters to be in one vector
grad = [X_grad(:); Theta_grad(:)];

end
