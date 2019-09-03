function [pcaResult, whitenResult, result] = Exercise5_3(varargin)

ip = inputParser;

% specify the random seed
rng(0);
% generate the data matrix
data = randn(2000, 2) * [2 1; 1 2];
% draw the data on the figure
subplot(2,2,1);
scatter(data(:, 1), data(:, 2));
title("(1)origin data");
% begin to perform the pca (PS: the (a) question)

% transform the matrix by reducing the mean value
meanVector = mean(data);
data = data - meanVector;
% calculate the cov matrix (PS: we use the X'X to calculate the two eigenvalue, we can prove that
% the first min(n, m) eigenvalue of X'X equal to those ones of the XX')
% covMatrix = X' * X;
% procude the svd to get the singularity value and right singu vectors
[~, D, VT] = svd(data);
singularityValues = diag(D);
eigenValues = singularityValues .* singularityValues;
% the eigenvectors are colum of VT
% result = meanVector + data * VT(1, :)' * VT(1, :) + data * VT(2, :)' * VT(2, :);
% PS: the more general representation(because data = [x1, x2, x3 ....]'&& VT = [v1, v2])
result = meanVector + data * VT * VT';
pcaResult = data*VT;
subplot(2,2,2);
scatter(pcaResult(:, 1), pcaResult(:, 2), 'r');
title("(2)data after PCA operation");
% use this line to avoid the visual equation between axis x and axis y
axis equal
% perform the whitening transform
w = diag(power(eigenValues, -1/2));
% PS: if we perform the norm operation, we could get the same figure as the
% picture in the book  i.e. the equation w = diag(power(eigenValues, -1/2)/norm(power(eigenValues, -1/2)));
whitenResult = data*VT*w;
subplot(2,2,3);
scatter(whitenResult(:, 1), whitenResult(:, 2), 'p');
title("(3) whitening transform");
subplot(2,2,4);
scatter(result(:, 1), result(:, 2), 'r');
title("(4) the approximation data");
