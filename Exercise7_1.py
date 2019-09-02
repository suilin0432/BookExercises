from libsvm import *
# PS: 因为我在libsvm的util的文件里面引入了 svmutil svm commonutil 所有内容所以
# 直接这么写就行了
import os

'''
构建库文件的使用类
'''
class Exercise7_1(object):
    def __init__(self):
        self.trainLabel = []
        self.trainData = []
        self.testLabel = []
        self.testData = []
        # 下面是默认的各个参数值
        self.cost = 1
        self.svmType = 0
        self.kernelType = 2
        self.gamma = 1
        self.degree = 3
        self.coef0 = 0

    def setCost(self, cost):
        assert cost > 0
        self.cost = cost

    def setSvmType(self, svmType):
        assert svmType in [0, 1, 2, 3, 4]
        self.svmType = svmType

    def setKernelType(self, kernelType):
        assert kernelType in [0, 1, 2, 3, 4]
        self.kernelType = kernelType

    def setGamma(self, gamma):
        assert gamma > 0
        self.gamma = gamma

    def setDegree(self, degree):
        assert degree > 0
        self.degree = degree

    def setCoef0(self, coef0):
        self.coef0 = coef0

    def train(self):
        # 返回模型 model
        assert len(self.trainLabel) > 0 and len(self.trainData) > 0 and len(self.trainLabel) == len(self.trainData)
        # svmType kernelType degree gamma coef0 cost
        param = "-s {0} -t {1} -d {2} -g {3} -r {4} -c {4}".format(self.svmType, self.kernelType, self.degree, self.gamma, self.coef0, self.cost)
        model = svm_train(self.trainLabel, self.trainData, param)
        return model

    def test(self, model):
        assert isinstance(model, svm_model)
        assert len(self.testLabel) > 0 and len(self.testData) > 0 and len(self.testLabel) == len(self.testData)
        pLabel, pAcc, pVal = svm_predict(self.testLabel, self.testData, model)
        return pLabel, pAcc, pVal

    def loadTrainDataFromFile(self, filePath):
        assert os.path.exists(filePath)
        self.trainLabel, self.trainData = svm_read_problem(filePath)
        assert len(self.trainLabel) == len(self.trainData)
        assert len(self.trainLabel) > 0

    def loadTestDataFromFile(self, filePath):
        assert os.path.exists(filePath)
        self.testLabel, self.testData = svm_read_problem(filePath)
        assert len(self.testLabel) == len(self.testData)
        assert len(self.testLabel) > 0

    def setTrainData(self, trainLabel, trainData):
        assert len(trainLabel) == len(trainData)
        assert len(trainLabel) > 0
        self.trainLabel, self.trainData = trainLabel, trainData

    def setTestData(self, testLabel, testData):
        assert len(testLabel) == len(testData)
        assert len(testLabel) > 0
        self.testLabel, self.testData = testLabel, testData


trainClass = Exercise7_1()
trainClass.loadTrainDataFromFile("./svmguide1.txt")
trainClass.loadTestDataFromFile("./svmguide1.t")
#
model = trainClass.train()

'''
下面是教程中的使用的例子说明 注释掉了
'''
'''
# 读取数据
# PS: svm_read_problem 还有一个参数是 return_scipy = False 即并不默认返回scipy格式的数据
yTrain, xTrain = svm_read_problem("./svmguide1.txt")
yTest, xTest = svm_read_problem("./svmguide1.t")
# for i in range(len(x)):
#     print(x[i], y[i])
'''

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
    13. -wi weight : 在 C-SVC 中设置类别 i 的参数 C 为 weight * C (默认是 1) (PS: 在处理类别不平衡的数据的时候有用)
    14. -v n : n-fold cross validation mode (要求 n >= 2)
    15. -q : quite mode
'''
'''
m = svm_train(yTrain, xTrain, '-c 4') # 因此这里 -c 4 指的是松弛的那个参数
p_label, p_acc, p_val = svm_predict(yTest, xTest, m)
print(p_label)
# p_acc 中包含三个参数: 1. accuracy 2. mean-squared error 3. squared correlation coefficient(for regression)
# ps: 分类问题的均方误差就是 错误率(因为判断错误的时候得到的误差的平方值总是1)
print(p_acc)
print(p_val)
'''
