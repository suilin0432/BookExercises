function [] = Exercise5_2(varargin)

ip = inputParser;

% rand('seed', 0)
% the value which will be added at the generated random data
avg = [1 2 3 4 5 6 7 8 9 10];
scaleList = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001];

for i = 1:length(scaleList)
    scale = scaleList(i);
    rng(0)
    % PS: the document says that nowadays we should use rng now to assign the random
    % seed instead of using rand('seed', 0)

    % generate the random data
    data = randn(5000, 10)+repmat(avg*scale, 5000, 1);
    %　calcute the mean value of data matrix
    m = mean(data);
    % normalize the m vector
    m1 = m / norm(m);
    % perform the PCA (without centering)
    [~, S, V] = svd(data);
    % change the diag matrix S to the vector which contains all the diag values
    S = diag(S);
    % get the first eigenvector
    e1 = V(:, 1);

    % perform the PCA (with centering)
    newdata = data - repmat(m, 5000, 1);
    [U, S, V] = svd(newdata);
    S = diag(S);
    new_e1 = V(:, 1);

    % find the relationship between first eigenvector(without centering) and
    % the average vector
    avg = avg - mean(avg);
    avg = avg / norm(avg);
    e1 = e1 - mean(e1);
    e1 = e1 / norm(e1);
    new_e1 = new_e1 - mean(e1);
    new_e1 = new_e1 / norm(new_e1);
    corr1 = avg * e1;
    corr2 = e1' * new_e1;
    fprintf("scale: %f corr1: %f corr2: %f\n", scale, corr1, corr2);
end

% e1'
% new_e1'
% avg
% m1
% (new_e1 - e1)'


