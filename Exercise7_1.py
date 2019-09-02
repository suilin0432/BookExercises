from libsvm import *
# PS: 因为我在libsvm的util的文件里面引入了 svmutil svm commonutil 所有内容所以
# 直接这么写就行了


'''
下面是教程中的使用的例子说明
'''

# 读取数据
yTrain, xTrain = svm_read_problem("./svmguide1.txt")
yTest, xTest = svm_read_problem("./svmguide1.t")
# for i in range(len(x)):
#     print(x[i], y[i])

'''
svm_train 的 配置参数说明:
    1. -s svm_type : 设置svm的类别 (默认是 0)
        0 -- C-SVC (多类别分类)
        1 -- nu-SVC (多类别分类)
        2 -- one-class SVM
        3 -- epsilon-SVR (回归)
        4 -- nu-SVR(regression)
    2. -t kernel_type : 设置核方法类别 (默认是 2)
        0 -- linear
        1 -- polynomial
        2 -- Gaussian
        3 -- sigmoid
        4 -- precomputed kernel
    3. -d degree : 设置核方法中的d参数(应该是多项式) (默认是 3)
    4. -g gamma : 应该是高斯核和多项式核函数之类的那个在前面的乘的系数 (默认是 1/feature的个数)
    5. -r coef0 : (应该是多项式核函数的那个括号里的+c) (默认是 0)
    6. -c cost : 设置C-SVC，epsilon-SVR, nu-SVR 中的参数C(那个加入了松弛条件之后的惩罚项的系数) (默认是 1)    
    7. -n nu : 设置nu-SVC, one-class SVM, nu-SVR 的参数 nu (默认是 0.5)
    8. -p epsilon : 设置epsilon-SVR中的 epsilon参数 (默认是 0.1)
    9. -m cachesize : 设置cache 大小(MB) (默认是 100)
    10. -e epsilon : 设置终止的条件 (默认是 0.001)
    11. -h shrinking : 设置是否使用shrinking heuristics, 0 or 1 (默认为 1)
    12. -b probability_estimates : 是否使用SVC/SVR模型去进行概率估计 0 or 1 (默认为 0)
    13. -wi weight : 在 C-SVC 中设置类别 i 的参数 C 为 weight * C (默认是 1)
    14. -v n : n-fold cross validation mode (要求 n >= 2)
    15. -q : quite mode
'''
m = svm_train(yTrain, xTrain, '-c 4') # 因此这里 -c 4 指的是松弛的那个参数
p_label, p_acc, p_val = svm_predict(yTest, xTest, m)
print(p_label)
# p_acc 中包含三个参数: 1. accuracy 2. mean-squared error 3. squared correlation coefficient(for regression)
# ps: 分类问题的均方误差就是 错误率(因为判断错误的时候得到的误差的平方值总是1)
print(p_acc)
print(p_val)
