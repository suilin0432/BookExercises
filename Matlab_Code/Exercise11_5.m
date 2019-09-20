function [integralImage, getSumFunction] = Exercise11_5(A)
% The implement of integral image

% Begin to calculate the integralImage
B = zeros(size(A));
B(1, 1) = A(1, 1);
for i = 2:size(A, 2)
    B(1, i) = B(1, i-1) + A(1, i);
end
for i = 2:size(A, 2)
    B(i, 1) = B(i-1, 1) + A(i, 1);
end
for i = 2:size(A, 1)
    for j = 2:size(A, 2)
        B(i, j) = B(i-1, j) + B(i, j-1) - B(i-1, j-1) + A(i, j);
    end
end
integralImage = B;
% set the function
function [result] = SumFunction(A, x1, y1, x2, y2)
if x1>1 && y1>1
    result = A(x2, y2) - A(x1-1, y2) - A(x2, y1-1) + A(x1-1, y1-1);
elseif x1 == 1 && y1 > 1
    result = A(x2, y2) - A(x2, y1-1);
elseif y1 == 1 && x1 > 1
    result = A(x2, y2) - A(x1-1, y2);
else
    result = A(x2, y2)
end
end
getSumFunction = @(A, x1, y1, x2, y2) SumFunction(A, x1, y1, x2, y2);
end
    