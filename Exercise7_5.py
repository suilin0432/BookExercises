# PS: using the LibLinear software (using dual coordinate descent method)
from svmliblinear import *
# 同libsvm在__init__.py中引入了所有的liblinear的相关文件，因此直接这么写就可以
import time

t1 = time.time()
trainLabel, trainData = svm_read_problem("./Exercise7_5_Data/rcv1_train.binary")
t2 = time.time()
# 设置训练参数为: Cost = 4, Cross Validation = 5
m = train(trainLabel, trainData, '-c 4 -v 5')
t3 = time.time()
print("The time of loading data is {0}.\n The time of training with 4-fold-cross-validation is {1}.".format(t2-t1, t3-t2))
