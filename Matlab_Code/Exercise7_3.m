function [totalABError, largestABError, relativeError, largestRError] = Exercise7_3()

% Data generate
x = 0.01 : 0.01 : 1;
y = 0.01 : 0.01 : 1;
% Error Calculate anonymous function
absoluteErrorCalculate = @(x, y) abs(x-y);
relativeErrorCalculate = @(x, y) abs((x-y)/x);
% Kernel Calculate anonoymous function
HIKernel = @(x, y) min(x, y);
powerMeanKernel = @(x, y, p) power((power(x, p)+power(y, p))/2, 1/p);
% the result variation
largestABError = 0;
totalABError = 0;
relativeError = 0;
largestRError = 0;
% calculate the error
param = -32;
for i = 1 : length(x)
    % actually, we can prove that the value is accurate enough, so we could
    % emit the following several lines
%     a = HIKernel(x(i), x(i));
%     b = powerMeanKernel(x(i), x(i), param);
%     currentABError = absoluteErrorCalculate(a, b);
%     totalABError =  totalABError + currentABError;
%     largestABError = max(largestABError, currentABError);
%     relativeError = relativeError + relativeErrorCalculate(a, b);
    for j = i+1 : length(x)
        a = HIKernel(x(i), x(j));
        b = powerMeanKernel(x(i), x(j), param);
        currentABError = absoluteErrorCalculate(a, b);
        totalABError =  totalABError + currentABError;
        largestABError = max(largestABError, currentABError);
        relativeError = relativeError + relativeErrorCalculate(a, b);
        largestRError = max(largestRError, relativeErrorCalculate(a, b));
    end
end
fprintf("The totalAbsoluteError is %f.\nThe largestAbsoluteError is %f.\nThe totalrelativeError is %f.\nThe largestRelativeError is %f.\n",totalABError, largestABError, relativeError, largestRError);

        