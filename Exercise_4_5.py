import numpy as np

class Exercise_4_5(object):
    def __init__(self, labelList=[], scoreList=[]):
        assert len(labelList)== len(scoreList)
        if len(scoreList) == 0:
            self.labelList = [1, 2, 1, 1, 2, 1, 2, 2, 1, 2]
            self.scoreList = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        else:
            self.labelList = labelList
            self.scoreList = scoreList
        self.length = len(self.scoreList)
        # 先将labelList和scoreList一一对应然后根据score进行sort
        self.pairList = []
        for i in range(len(self.labelList)):
            self.pairList.append((self.labelList[i], self.scoreList[i]))

        self.pairList = sorted(self.pairList, key=lambda x:x[1], reverse=True)
        self.originPrecisionList = [1.0]
        self.originRecallList = [0.0]
        self.precisionList = [1.0]
        self.recallList = [0.0]

    def ExercisePRCalculate(self):
        self.precisionList = self.originPrecisionList.copy()
        self.recallList = self.originRecallList.copy()
        # Precision = TP/TP+FP Recall = TP/TP+FN
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in self.labelList:
            if i == 1:
                FN += 1
            else:
                TN += 1
        for i in range(self.length):
            if self.labelList[i] == 1:
                TP += 1
                FN -= 1
            else:
                FP += 1
                TN -= 1
            self.precisionList.append(TP/(TP+FP))
            self.recallList.append(TP/(TP+FN))
        # test code
        # print(self.precisionList)
        # print(self.recallList)

    def getAucPR(self):
        AucScore = 0
        for i in range(self.length):
            AucScore += (self.recallList[i+1] - self.recallList[i]) * \
                        (self.precisionList[i+1] + self.precisionList[i]) / 2
        return AucScore

    def getAP(self):
        APScore = 0
        for i in range(self.length):
            APScore += (self.recallList[i+1] - self.recallList[i]) * \
                        self.precisionList[i+1]
        return APScore

    def setLabelList(self, labels):
        assert len(labels) == self.length
        self.labelList = labels

    def setScoreList(self, scores):
        assert len(scores) == self.length
        self.scoreList = scores

    def setScoreAndLabel(self, scores, labels):
        assert len(scores) == len(labels)
        assert len(scores) > 0
        self.scoreList = scores
        self.labelList = labels
        self.length = len(scores)

a = Exercise_4_5()
a.ExercisePRCalculate()
print("AUC Score is: {0}\n AP Score is: {1}". format(a.getAucPR(), a.getAP()))
print("swap the 9th item and the 10th item")
a.labelList[8] = 2
a.labelList[9] = 1
a.ExercisePRCalculate()
print("After swapping the label, AUC Score is: {0}\n AP Score is: {1}". format(a.getAucPR(), a.getAP()))