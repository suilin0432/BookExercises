from libsvm import *
from scipy import sparse
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
        self.setDefaultParams()

    def setDefaultParams(self):
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
        param = "-s {0} -t {1} -d {2} -g {3} -r {4} -c {5}".format(self.svmType, self.kernelType, self.degree, self.gamma, self.coef0, self.cost)
        # print(param)
        model = svm_train(self.trainLabel, self.trainData, param)
        return model

    def predict(self, model):
        # 不知道为什么 isinstance(model, svm_model) 在这里是false 在ipython里命名就是True
        # assert isinstance(model, svm_model)
        assert str(type(model)) == "<class 'svm.svm_model'>"
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

    def getTrainData(self):
        return self.trainData

    def getTestData(self):
        return self.testData

    def getTrainLabel(self):
        return self.trainLabel

    def getTestLabel(self):
        return self.testLabel


if __name__=='__main__':
    trainClass = Exercise7_1()
    trainClass.loadTrainDataFromFile("./svmguide1.txt")
    trainClass.loadTestDataFromFile("./svmguide1.t")
    # 1. the origin version -- C = 1 and kernelType = 2 (RBF/Gaussian kernel)
    model1 = trainClass.train()
    pLabel1, pAcc1, pVal1 = trainClass.predict(model1)
    print("The accuracy of the first model (C = 1 and kernelType = 2) is {0}%.".format(pAcc1[0]))
    # 2. 使用svm_scale函数 并没有找到svm_scale这个函数... 是外面的一个运行文件时svm_scale
    # 命令1: ./svm-scale -l -1 -u 1 -s ../ExerciseCode_Python/scaleParam ../ExerciseCode_Python/svmguide1.txt > ../ExerciseCode_Python/svmguide1_.txt
    # 命令2: ./svm-scale -r ../ExerciseCode_Python/scaleParam ../ExerciseCode_Python/svmguide1.t > ../ExerciseCode_Python/svmguide1_.t
    trainClassScale = Exercise7_1()
    trainClassScale.loadTrainDataFromFile("./svmguide1_.txt")
    trainClassScale.loadTestDataFromFile("./svmguide1_.t")
    model2 = trainClassScale.train()
    pLabel2, pAcc2, pVal2 = trainClassScale.predict(model2)
    print("The accuracy of the second model (C = 1 and kernelType = 2 and datas are scaled) is {0}%.".format(pAcc1[0]))

    # 3. 使用linear kernel代替 RBF kernel 即使用 -t 参数为0
    trainClass.setKernelType(0)
    model3 = trainClass.train()
    pLabel3, pAcc3, pVal3 = trainClass.predict(model3)
    print("The accuracy of the third model (C = 1 and kernelType = 0) is {0}%.".format(pAcc3[0]))

    # 4. 将 Cost = 1000 然后使用 RBF kernel
    trainClass.setDefaultParams()
    trainClass.setCost(1000)
    model4 = trainClass.train()
    pLabel4, pAcc4, pVal4 = trainClass.predict(model4)
    print("The accuracy of the forth model (C = 1000 and kernelType = 2) is {0}%.".format(pAcc4[0]))

    # 5. 使用easy.py去找超参数
    # 使用命令: python3 easy.py ../../ExerciseCode_Python/svmguide1.txt ../../ExerciseCode_Python/svmguide1.t
    # 找到的超参数: c=8.0, g=2.0
    # 在这里试一下
    trainClass.setDefaultParams()
    trainClass.setCost(8)
    trainClass.setGamma(2)
    model5 = trainClass.train()
    pLabel5, pAcc5, pVal5 = trainClass.predict(model5)
    print("The accuracy of the forth model (C = 8 and gamma = 2 and without scale) is {0}%.".format(pAcc5[0]))

    trainClassScale.setDefaultParams()
    trainClassScale.setCost(8)
    trainClassScale.setGamma(2)
    model5_ = trainClassScale.train()
    pLabel5_, pAcc5_, pVal5_ = trainClassScale.predict(model5_)
    print("The accuracy of the forth model (C = 8 and gamma = 2 and with sacle) is {0}%.".format(pAcc5_[0]))


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
