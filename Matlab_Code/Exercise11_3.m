function [] = Exercise11_3()

% generate the dictionary with 300 items(with 150 dimensions)
% we will chose 40 dictionary items to generate example x and add some
% noise to x

% PS: rand: 生成均匀分布的伪随机数 randn: 生成标准正态分布的伪随机数

p = 150; % dimension
k = 300; % dictionary size
D = randn(p,k); % dictionary PS: the item of D is arranged as column
% normalize dictionary item
for i = 1:k
    D(:,i) = D(:,i)/norm(D(:,i));
end

% generate the example x (by using 40 dimensions of D)
truealpha = zeros(k, 1);
% random 40 items and assign randomize weights on corresponding entry
truealpha(randperm(k, 40)) = 30 * (rand(40, 1) - 0.5);
% add noise to x
noisealpha = truealpha + .1*randn(size(truealpha));
x = D * noisealpha;

% Exercise(a). using the ordinary least square optimal solution to solve the question
% if we use inv to calulate A^-1 all the time, we might get a bad result
% when D'*D is singular, so we could use pinv to calculate the
% pseudo-inverse
DD = D'*D;
[height, width] = size(D);
if height >= width
    alpha1 = inv(DD)*D'*x;
else
    alpha1 = pinv(DD)*D'*x;
end
fprintf("Exercise(a):\n");
fprintf("The distance between the truealpha and the alpha calculated by ordinary least square optimal solution is %f(norm2), %f(norm1).\n", norm(alpha1-truealpha), norm(truealpha-alpha1, 1));
threshold = 0.1;
count1 = 0;
for i = 1:length(alpha1)
    if abs(alpha1(i)) <= threshold
        count1 = count1 + 1;
    end
end
fprintf("When we consider the zero entries as entries whose values are less than %f, there are %d zero entries.\n", threshold, count1);
% we could see that there are nearly none zero entries in the alpha array
% calculated by ordinary least square optimal solution.

% Exercise(b). using the FISTA to get the solution

% calculate the maximum eigenvalue of D'D
L = eigs(D'*D, 1);
% initialize the parameters
stopStep = 100;
t = 1;
alpha0 = alpha1;
beta = alpha0;
LAMBDA = 1;
% calculate the D'*x and D'*D first to accurate the calculation
DX = D'*x;
DTD = D'*D;

% begin to calculate util run stopStep times
for i = 1:stopStep
    % solve the argmin problem
    alpha_ = zeros(size(beta));
    y = beta - 1/L*(DTD*beta-DX);
    for j = 1:length(alpha_)
        alpha_(j) = max(y(j) - LAMBDA/2, 0);
        if y(j)<0
            alpha_(j) = -alpha_(j);
        end
    end
    % update t
    t_ = (1+sqrt(1+4*t*t)) / 2;
    beta = alpha_ + (t-1)/t_*(alpha_-alpha0);
    alpha0 = alpha_;
    t = t_;
end
fprintf("\nExercise(b):\n");
count2 = 0;
for i = 1:length(alpha0)
    if abs(alpha0(i)) < 1e-3
        count2 = count2 + 1;
    end
end
fprintf("We could find %d entries with zero value(<1e-3).\n", count2);
fprintf("The distacne between alpha0 and truealpha is %f(norm2), %f(norm1).\n", norm(truealpha-alpha0), norm(truealpha-alpha0, 1));
% It's true that we could get a sparsity array, however, the norm2-distance
% between alpha0 and truealpha is larger than alpha1 and truealpha;
% Em... the norm1-distance of alpha0 and truealpha is smaller.

% Exercise(c): 
fprintf("\nExercise(c):\n");
LAMBDALIST = [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.5, 1, 1.5, 2, 5, 10, 1e2, 1e3, 1e4];
% em... I just copy part of the code above... Don't mind ... 
for i = 1:length(LAMBDALIST)
    t = 1;
    alpha0 = alpha1;
    beta = alpha0;
    LAMBDA = LAMBDALIST(i);
    for k = 1:stopStep
        % solve the argmin problem
        alpha_ = zeros(size(beta));
        y = beta - 1/L*(DTD*beta-DX);
        for j = 1:length(alpha_)
            alpha_(j) = max(y(j) - LAMBDA/2, 0);
            if y(j)<0
                alpha_(j) = -alpha_(j);
            end
        end
        % update t
        t_ = (1+sqrt(1+4*t*t)) / 2;
        beta = alpha_ + (t-1)/t_*(alpha_-alpha0);
        alpha0 = alpha_;
        t = t_;
    end
    count_ = 0;
    for j = 1:length(alpha0)
        if abs(alpha0(j))<1e-3
            count_ = count_ + 1;
        end
    end
    fprintf("When λ = %f, we have %d zero entries, and the distance between truealpha and alpha0 is %f(norm2), %f(norm1). \n", LAMBDA, count_, norm(alpha0-truealpha), norm(alpha0-truealpha, 1));
end

% We could see when λ increases, the number of zero entries becomes larger.
    
    
    




