import random
import numpy as np
import time

class Exercise_4_8(object):
    def __init__(self, dataset=[]):
        if(len(dataset) == 0):
            self.dataset = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        else:
            self.dataset = dataset
        self.length = len(self.dataset)

    def exercise1(self, times=10):
        zeroTwoTimes = 0
        oneOneTimes = 0
        for i in range(times):
            time.sleep(0.0001)
            count = 0
            data = [i for i in range(self.length)]
            np.random.shuffle(data)
            for j in range(self.length//2-1):
                if self.dataset[data[j]] == 1:
                    count += 1
            if count == 0 or count == 2:
                zeroTwoTimes += 1
            else:
                oneOneTimes += 1
        # for i in range(times):
        #     count = 0
        #     data = self.dataset.copy()
        #     random.shuffle(data)
        #     for j in range(self.length//2-1):
        #         if data[j] == 1:
        #             count += 1
        #     if count == 0 or count == 2:
        #         zeroTwoTimes += 1
        #     else:
        #         oneOneTimes += 1
        return zeroTwoTimes, oneOneTimes

a = Exercise_4_8()
t = 100
times = 1000
two, one = a.exercise1(times=t)
print("(0,2) appears {0} times, (1, 1) appears {1} times".format(two, one))
twoSum = 0
oneSum = 0
for i in range(times):
    print(i)
    two, one = a.exercise1(times=t)
    twoSum += two
    oneSum += one
print("after {0} times, (0,2) appears {1} times, (1, 1) appears {2} times".format(times*10, twoSum, oneSum))


