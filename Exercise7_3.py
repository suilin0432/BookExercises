import Exercise7_1
from scipy import sparse
import os
import time

trainClass = Exercise7_1.Exercise7_1()
trainClass.loadTestDataFromFile("./test1.txt")
trainClass.loadTrainDataFromFile("./train1.txt")

# 1. all the superParameters are the default value
# in the Exercise7_3_Result.txt I have delete all median messages because of the size of the file
# it's not accurate at all and time-cost
t1 = time.time()
model1 = trainClass.train()
t2 = time.time()
pLabel1, pAcc1, pVal1 = trainClass.predict(model1)
t3 = time.time()
print("The accuracy is {0}%.".format(pAcc1[0]))
print("The timecost is:\n Training time: {0}, Testing time: {1}".format(t2 - t1, t3 - t2))

# 2. using easy.py
# It costs such a long time.
