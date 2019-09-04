import Exercise7_1
from scipy import sparse
import os
import time

trainClass = Exercise7_1.Exercise7_1()
trainClass.loadTestDataFromFile("./test1.txt")
trainClass.loadTrainDataFromFile("./train1.txt")
trainClass.setQuiet(True)

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
# It costs such a long time. I have trained for at least 6 hours.
# the output:
'''
Scaling training data...
WARNING: original #nonzeros 2258542
       > new      #nonzeros 9392885
If feature values are non-negative and sparse, use -l 0 rather than the default -l -1
Cross validation...
  
Best c=8.0, g=0.001953125 CV rate=51.0231
Training...
Output model: train1.txt.model
Scaling testing data...
WARNING: original #nonzeros 2976002
       > new      #nonzeros 12443253
If feature values are non-negative and sparse, use -l 0 rather than the default -l -1
Testing...
Accuracy = 53.3632% (1071/2007) (classification)
Output prediction: test1.txt.predict
'''

# 3. set the parameter as the value given by easy.py, but without scaling the data
trainClass.setCost(8)
trainClass.setGamma(0.001953125)
model2 = trainClass.train()
pLabel2, pAcc2, pVal2 = trainClass.predict(model2)
print("The accuracy is {0}%.".format(pAcc2[0]))