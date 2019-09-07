function [] = Exercise8_5()

% The Code of the question (a).
% Function: generate a two-dimensional normal density. PS: It's the GT
% value calculated by the GT distribution.

% covariance matrix
iSigma = inv([2 1; 1 4]);
pts = -5:0.1:5;
l = length(pts);
GT = zeros(l); % the size of the generated matrix is 101*101.
for i = 1:l
    for j = 1:l
        temp = [pts(i) pts(j)];
        GT(i, j) = exp(-0.5*temp*iSigma*temp');
    end
end
% normalize GT in order to let it be a valid pdf
GT = GT / sum(GT(:));

% Exercise (b):
% PS: 1. Both the two normal random variables have zero mean, because the pdf
% uses temp instead of (temp - meanValue)
% 2. The two normal random variables could have different standard deviations

% Task: Thinking that the two variables are independent, so we could use
% the product of their pdf to approximate the nondiagonal complex Gaussian
% density

% Solution: In this condition, the iSigma matrix is diagnol. There are no 
% relationships between a & b.
iSigma2 = inv([2 0; 0 4]);
predict2 = zeros(l);
recordX = zeros(1, l*l);
recordY = zeros(1, l*l);
for i = 1:l
    for j = 1:l
        recordX((i-1)*l+j) = pts(i);
        recordY((i-1)*l+j) = pts(j);
        temp = [pts(i) pts(j)];
        predict2(i, j) = exp(-0.5*temp*iSigma2*temp');
    end
end
predict2 = predict2 / sum(predict2(:));

% plot the figure to compare the GT pdf and the approximate pdf.

GTR = reshape(GT, [1, l*l]);
subplot(2,1,1);
predict2R = reshape(predict2, [1, l*l]);
plot3(recordX, recordY, GTR, 'b');
hold on;
plot3(recordX, recordY, predict2R, 'r');
legend("GT pdf", "approximate pdf");
Error1 = 1 - sum(min(GT(:),predict2(:)));
fprintf("The error1 is %f.\n", Error1);

% We could see that the two pdf is pretty similar.

% Exercise (c):
% Task: Try different values of variances of a & b. and try to find the
% best pair which has the min loss
minError = 1;
minV1 = 0;
minV2 = 0;
varList = 0.05:0.05:5;
varLength = length(varList);
predict3R = zeros(l);
for i = 1:varLength
    for j = 1:varLength
        iSigma3 = inv([varList(i) 0; 0 varList(j)]);
        predict3 = zeros(l);
        for m = 1:l
            for n = 1:l
                temp = [pts(m) pts(n)];
                predict3(m, n) = exp(-0.5*temp*iSigma3*temp');
            end
        end
        predict3 = predict3 / sum(predict3(:));
        Error = 1 - sum(min(GT(:),predict3(:)));
        if minError > Error
            minError = Error;
            predict3R = reshape(predict3, [1, l*l]);
            minV1 = varList(i);
            minV2 = varList(j);
        end
    end
end
fprintf("The minError is %f, the variance of a is %f and the variance of b is %f.\n", minError, minV1, minV2);

% plot the pdf with the minError
subplot(2,1,2);
plot3(recordX, recordY, GTR, 'b');
hold on;
plot3(recordX, recordY, predict3R, 'r');
legend("GT pdf", "approximate pdf");
% we could see the minError is about 0.126675. variance of a is 2 variance 
% of b is 3. It seems like not better than the former one. I think it might
% be caused by the range of the variance value, so when I change the range
% from 0.05 ~ 3 to 0.05 ~ 4 It has the minError 0.113768 when the variance
% of a is 1.85 and the variance of b is 3.75. And when I change the range
% upperbound to 5, we could get the minError is also euqal to 0.113768 with
% the variance of a is and the variance of b is .
