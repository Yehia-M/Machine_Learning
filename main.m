%% % Load data
load('ex8_movies.mat');
%% 
% Load movvie list
movieList = loadMovieList();
% Initialize new ratings
myratings = init_myratings();
%% Print the new ratings
for i = 1:length(myratings)
    if myratings(i) > 0 
        fprintf('Rated %d for %s\n', myratings(i), movieList{i});
    end
end
%% 
%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
%  Add new ratings to the data matrix
Y = [myratings Y];
R = [((myratings ~= 0)+0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj','on','MaxIter',100);
% Set Regularization
lambda = 10;
%% 
%intital run
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features,lambda)), initial_parameters, options);
%% 
%Run another 100 iterations
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features,lambda)), theta, options);
%% % Unfold the returned theta back into X and Theta
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);
%% 
%Prediction for the new user
p = X * Theta';
my_predictions = p(:,1) + Ymean;
%% 
[r, ix] = sort(my_predictions,'descend');
for i=1:50
    j = ix(i);
    if i == 1
        fprintf('\nTop recommendations for you:\n');
    end
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end

%% Save X, Theta and Ymean for future use
save('X.mat','X');
save('Theta.mat','Theta');
save('Ymean.mat','Ymean');