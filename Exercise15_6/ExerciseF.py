import torch
from torch import nn
from DataReader import DataReader
from MNISTDataSet import MNISTDataSet
from torch.utils.data import DataLoader
from SKNet import SKNet
import time
import os
import matplotlib.pyplot as plt
from multiprocessing import Process

def oneTime(name, cAlpha, lr, INDEX):
    print("{}开始进程".format(name))
    dataReader = DataReader()
    trainImages = dataReader.loadTrainImages()
    trainLabels = dataReader.loadTrainLabels()
    testImages = dataReader.loadTestImages()
    testLabels = dataReader.loadTestLables()
    net = SKNet(cAlpha=cAlpha)
    cuda = True
    if cuda:
        net = net.cuda()
    trainDataNumber = 10000
    # 采用的是 matconvnet 的配置
    batch_size = 100
    maxEpoch = 1
    lr = lr
    # PS: 很神奇的是用 Adam 的时候损失函数不收敛... 一点都不变化... 用 Adam 的时候降低学习率就好了...
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # print(trainImages.shape, trainLabels.shape, testImages.shape, testLabels.shape)
    trainDataset = MNISTDataSet(trainImages, trainLabels, trainDataNumber)
    testDataset = MNISTDataSet(testImages, testLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=32)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=True, num_workers=32)
    index = 0
    losses = []
    for epoch in range(maxEpoch):
        time1 = time.time()
        for i, data in enumerate(trainLoader):
            index += 1
            images, label = data
            # print(images.shape, label.shape)
            images = images.unsqueeze(1)
            images = images.float()
            if cuda:
                images = images.cuda()
                label = label.cuda()
            label = label.long()
            predict = net(images, False)
            loss = net.loss(predict, label)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss)
            optimizer.step()
            if index % 100 == 0:
                print(name, "iter {} loss: {}".format(index, loss))
        print(name, "time cost of epoch {}: {}".format(epoch, time.time() - time1))

    count = 0
    correct = 0
    net = net.eval()
    for i, data in enumerate(testLoader):
        count += 1
        if count % 1000 == 0:
            print("current: {}/{}".format(correct, count))
        image, label = data
        if cuda:
            image = image.cuda()
            label = label.cuda()
        image = image.float()
        image = image.unsqueeze(1)
        label = label.long()
        predict = net(image)
        maxValue, maxIndex = torch.max(predict, 1)
        maxIndex = maxIndex.long()
        if maxIndex == label:
            correct += 1
    print("lr: {} accuracy rate: {}".format(lr, correct / count))

    lossesList[INDEX] = losses
    accuracyList[INDEX] = correct/count


if __name__ == "__main__":
    # PS: 每个跑两次
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    lossesList = [[] for i in range(30)]
    accuracyList = [0 for i in range(30)]
    nameList = [0 for i in range(30)]
    cAlphaList = [0.2]
    # cAlphaList = [2, 1.5, 1, 0.5, 0.2]
    lrList = [0.001, 0.002, 0.1]
    count = 0
    for cAlpha in cAlphaList:
        paramList = []
        for lr in lrList:
            for i in range(2):
                paramList.append({
                    "name":"cAlpha: {} lr: {}".format(cAlpha, lr),
                    "cAlpha": cAlpha,
                    "lr": lr,
                    "INDEX": count
                })
                nameList[count] = "cAlpha: {} lr: {}".format(cAlpha, lr)
                count += 1
        processList = []
        for i in paramList:
            p = Process(target=oneTime, kwargs=i)
            processList.append(p)
        for p in processList:
            p.start()
        for p in processList:
            p.join()
    count = 0
    for i in range(30):
        count += 1
        plt.subplot(5, 6, count)
        losses = lossesList[count-1]
        accuracy = accuracyList[count-1]
        name = nameList[count-1]
        print("name: {}, accuracy: {}".format(name, accuracy))
        plt.plot([i for i in range(len(losses))], losses)
        plt.title("{}, accuracy:{}".format(name, accuracy))
    plt.show()

