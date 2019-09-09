from svmliblinear import *
import math
import time

# Read the Training Data and Testing Data
time1 = time.time()
trainLabel, trainData = svm_read_problem("Exercise9_6_Data/mnist")
testLabel, testData = svm_read_problem("Exercise9_6_Data/mnist.t")
time2 = time.time()
print("using {} s to load the data".format(time2 - time1))
# Perform sqrt transformation on the Data Set
time3 = time.time()
sqrtTrainData = []
sqrtTestData = []
for i in range(len(trainData)):
    DICT = {}
    for index, data in trainData[i].items():
        DICT[index] = math.sqrt(data)
    sqrtTrainData.append(DICT)
for i in range(len(testData)):
    DICT = {}
    for index, data in testData[i].items():
        DICT[index] = math.sqrt(data)
    sqrtTestData.append(DICT)
time4 = time.time()
print("using {} s to perform sqrt data transformation".format(time4 - time3))
# Start training(using the default parameters)
time5 = time.time()
model1 = train(trainLabel, trainData)
p_label1, p_acc1, p_val1 = predict(testLabel, testData, model1)
print("The accuracy is {} when we perform the training process without any transform of trainData".format(p_acc1[0]))

model2 = trainData(trainLabel, sqrtTrainData)
p_label2, p_acc2, p_val2 = predict(testLabel, sqrtTestData, model2)
print("The accuracy is {} when we perform the training process with sqrt transform of trainData".format(p_acc2[0]))


time6 = time.time()
print("using {} s to training".format(time6 - time5))
