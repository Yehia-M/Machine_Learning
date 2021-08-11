%% % The following command loads the dataset
load('ex8data1.mat');

% Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
%% 
%  Estimate mu and sigma2
[mu, sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2);
%% 
%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
%% 
pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);

%% %  Find the outliers in the training set and plot
outliers = find(p < epsilon);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off
%% Bigger Dataset
load('ex8data2.mat');

[mu, sigma2] = estimateGaussian(X); %Find mean and s.d of the new dataset
p = multivariateGaussian(X, mu, sigma2); %Training set 
pval = multivariateGaussian(Xval, mu, sigma2); %Cross-validation set
[epsilon, F1] = selectThreshold(yval, pval); %Find the best threshold

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);

%% For new predictions
% X_Test_Example = matrix [1 x 11(no. of features)]
% prediction = multicariateGaussian(X_Test_Example,mu,sigma2) where my and
% sigma2 are already known parameters
% if predicion < epsilon, then example is considered outlier
