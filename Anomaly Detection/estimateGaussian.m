function [mu,sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector

[m, n] = size(X);                            %No. of examples X No. of features
mu = 1/m * sum(X);                           %Mean of all features of X = a NX1 Matrix 
sigma2 = 1/m * sum((X-repmat(mu,[m,1])).^2); %S.D of all features of X = a NX1 Matrix 
end
