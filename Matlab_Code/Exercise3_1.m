function [result, kdResult, errorRate] = Exercise3_1(varargin)
% This function is the code of Exercise3_1
% args:
%   pointCount: the number of generated data points
%

% set the default value
ip = inputParser;
validNumber = @(x) isnumeric(x) && isscalar(x) && x > 0;
ip.addParameter("x", []);
ip.addParameter("pointCount", 5000, validNumber);
ip.addParameter("numTrees", 1, validNumber);
ip.addParameter("maxNumComparisons", 6000, @(x) validNumber(x) || x == 0);
ip.parse(varargin{:});

ipResults = ip.Results;
pointCount = ipResults.pointCount;
numTrees = ipResults.numTrees;
maxNumComparisons = ipResults.maxNumComparisons;
x = ipResults.x;

% generate the data
if length(x) == 0
    fprintf("generating the data\n");
    x = rand(pointCount, 10);
end
MAX_INT = 2147483647;

% the code which is implemented from scratch
% PS: if we should not use functions such as sum and min which are
% linear-time-cost operation, we need to use loop to replace the functions

tic;
result = zeros(pointCount,2);
for i = 1:pointCount
    m = x(i, :) - x;
    m = m.^2; %每个数字进行平方处理
    m = sum(m, 2);
    m(i) = MAX_INT;
    [result(i,2), result(i,1)] = min(m);
%     fprintf("%d, %d\n", result(i,2), result(i, 1));
end
t1 = toc;

fprintf("the time cost of NN method: %f\n",t1);
% using kd-tree
kdTree = vl_kdtreebuild(x', 'NumTrees', numTrees);
tic;
[index, distance] = vl_kdtreequery(kdTree, x', x', 'NumNeighbors', 2, 'MaxComparisons', maxNumComparisons);
% 如果没有进行double转化的话　会是一个uint32数组　浮点精度全都没了...
% distance
kdResult = [double(index(2, :)'), distance(2, :)'];
t2 = toc;

fprintf("the time cost of kd-tree method: %f\n", t2);

%error rate compute
errorNum = 0;

for i = 1:pointCount
    if kdResult(i, 1) ~= result(i, 1)
        errorNum = errorNum + 1;
    end
end
errorRate = errorNum/pointCount;

fprintf("the error rate of the kd-tree is: %f\n", errorRate);
        

