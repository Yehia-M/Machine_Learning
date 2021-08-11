%% 
movieList = loadMovieList();
load('X.mat');
load('Theta.mat');
load('Ymean.mat');

%% 
%Prediction for Un user
Un = 1;
p = X * Theta';
my_predictions = p(:,Un) + Ymean;
%% 
[r, ix] = sort(my_predictions,'descend');
for i=1:10
    j = ix(i);
    if i == 1
        fprintf('\nTop recommendations for you:\n');
    end
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end