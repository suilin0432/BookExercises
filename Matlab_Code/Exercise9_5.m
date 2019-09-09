function [] = Exercise9_5()
% generate the data
x = -7:1:7;
xb = [x; ones(size(x))]; % put variable b and variable w together.
rng(0); % set the random seed
noise = rand(size(x))*0.2;
w = [0.3, 0.2];
z = w*xb + noise;

% Exercise(a): using ordinary least square estimation to estimate w and b
w_ = z*xb'*inv(xb*xb');
fprintf("Exercise(a):\n");
fprintf("We could get that the estimated w value is: %f, the estimated b value is: %f.\n", w_(1), w_(2));

% Exercise(b,c,d): performing the ridge regression which adds the term λ||β||^2
fprintf("\nExercise(b, c, d):\n");
LAMBDA = [0.01, 0.1, 1, 10, 100, 9.3];
LAMBDA = [LAMBDA 9:0.05:10];
eyeSize = size(xb, 1);
for i = 1:length(LAMBDA)
    ld = LAMBDA(i)
    w_ = z*xb'*inv(xb*xb'+ld*eye(eyeSize));
    fprintf("When λ equal to %f, the estimated w is: %f, estimated b is: %f.\n", ld, w_(1), w_(2));
end
% PS: When λ becomes larger and larger, the estimated w and b are becoming
% smaller. When λ is suitable enough(about 9～10), we could get a good
% value.

% And we could see both the value of estimated w an d the value of
% estimated b are predicted better than ordinary least square method.