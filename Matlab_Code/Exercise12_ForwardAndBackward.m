function [] = Exercise12_ForwardAndBackward()
% The code in page 15 of Chapter12, which is used to evaluted that the
% forward algorithm and the backward algorithm will get the same result.

iter = 1;
for iter = 1:1000
    N = 3; % State的状态数量
    % 初始化状态的分布
    Pi = rand(1, N);
    % PS: 原来的代码写的 1/sum(Pi); 这样产生的是一个scalar... 并不是分布, 后面虽然也不会影响验证结果
    % 但是不符合概率分布
    Pi = Pi / sum(Pi);
    % 测试代码
%     Pi = [0.2, 0.3, 0.5];
    % 状态转移矩阵A计算
    A = rand(N, N);
    A(1, 3) = 0; % 设置了1->的转移是不可能的
    for i = 1:N
        A(i, :) = A(i, :)/sum(A(i, :));
    end
    % 测试代码
%     A = [0.8, 0.1, 0.1; 0.2, 0.6, 0.2; 0.3, 0, 0.7];
    % Observation的可能数量
    M = 3;
    % 概率矩阵B(状态产生observation的概率矩阵)计算
    B = rand(N, M);
    for i = N
        B(i, :) = B(i, :)/sum(B(i, :));
    end
    % 测试代码
%     B = [0.3, 0.3, 0.4; 0.5, 0.2, 0.3; 0.1, 0.8, 0.1];
    % HMM链长度
    T = 5;
    % 随机产生的我们想要验证的输出 randi(imax, size1, size2); -> 即生成1*5的array
    % 其值会在1-3之间 代表着长度为5的HMM链的observation 我们要计算的就是产生这个链的概率
    O = randi(M, 1, T);
    
    % alpha 这里是记录所有alpha_t的值, 每一t位置都有N个值要计算, 其实可以改用(2,N)array
    Alpha = zeros(T, N);
    % beta 同 alpha
    Beta = ones(T, N);
    % 计算Alpha 
    %   ***TODO: 这里应该是 B(:, O(1)) 吧 原来代码的 O(1)直接将第一个output固定为了1
    %   虽然不会影响验证效果... 
%     Alpha(1, :) = Pi .* B(:, 1)';
    Alpha(1, :) = Pi .* B(:, O(1))';
    
    for t = 2:T
        Alpha(t, :) = (Alpha(t-1, :) * A) .* B(:, O(t))';
    end
    % 计算Beta
    for t = (T-1):-1:1
        Beta(t, :) = A * (B(:, O(t+1)) .* Beta(t+1, :)');
    end
    
    % 计算Gamma 这里是beta的最后一步计算, 巧妙的利用了Alpha(1, :)的值去计算, 但是多余的维度的计算是不必要的,
    % 所以改为Alpha(1, :).*Beta(1, :)更好吧 但是后面是要用到整个Gamma的...
    % 进一步 按照公式来说 Gamma(t, :) 是在 给定的Observation序列下 Qt = State_i 时候的概率
    Gamma = Alpha.*Beta;
%     Gamma = Alpha(1, :).*Beta(1, :);
    % 计算Forward以及Backward两种方式进行计算得到的概率
    p1 = sum(Alpha(end, :));
    p2 = sum(Gamma(1, :));
%     p2 = sum(Gamma);
    assert(abs(p1-p2)<1e-12);
    % 查看是否有 1->3 的我们不允许的情况出现
    [~, I] = max(Gamma');
    for i = 1:T-1
        if I(i) == 1 && I(i+1) == 3
            disp(['1-->3 at iteration ' num2str(iter) '!'])
            return
        end
    end
    % PS: 是可能会发生的... 就是当前最大的概率在 1时候 到下一次虽然1->3没有贡献,
    % 但是2->3以及3->3产生的贡献可能会超过给1->1,2->1,3->1的贡献和, 虽然产生概率比较小
    % eg: 当前概率分布 1:0.4 2:0.3 3:0.3 状态1最可能出现 
    % 当A为 0.8 0.2 0; 0 0.1 0.9; 0 0.1 0.9时候 显然 0.32 < 0.54 状态会变为 3 
    % 但是情况要比较少见而已, 可以观察发现出现异常的时候都是在 -> 1的概率比较小 ->3 的概率比较大的时候出现的 
end
    
    
    