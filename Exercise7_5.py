# PS: using the LibLinear software (using dual coordinate descent method)
import svmliblinear
import libsvm
# 同libsvm在__init__.py中引入了所有的liblinear的相关文件，因此直接这么写就可以
import time

# 可以发现两者效果类似，但是liblinear明显节省了很多时间，大概只需要几秒，而libsvm需要十几分钟

# training with liblinear
t1 = time.time()
trainLabel, trainData = svmliblinear.svm_read_problem("./Exercise7_5_Data/rcv1_train.binary")
t2 = time.time()
# 设置训练参数为: Cost = 4, Cross Validation = 5 使用 dual形式的L1-loss的SVC(即 3)
m = svmliblinear.train(trainLabel, trainData, '-s 3 -c 4 -v 5')
t3 = time.time()
print("The time of loading data is {0}.\n Training with liblinear:\nThe time of training with 5-fold-cross-validation is {1}.".format(t2-t1, t3-t2))

# training with libsvm
t4 = time.time()
modelLibSVM = libsvm.svm_train(trainLabel, trainData, '-t 0 -c 4 -v 5')
t5 = time.time()
print("Training with libsvm:\n The time of training with 5-fold-cross-validation is {0}.".format(t5-t4))