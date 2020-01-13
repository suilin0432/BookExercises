import torch
from torch import nn
from DataReader import DataReader
from MNISTDataSet import MNISTDataSet
from torch.utils.data import DataLoader
from SKNet import SKNet
import time
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataReader = DataReader()
    trainImages = dataReader.loadTrainImages()
    trainLabels = dataReader.loadTrainLabels()
    testImages = dataReader.loadTestImages()
    testLabels = dataReader.loadTestLables()
    net = SKNet(cAlpha=1)
    cuda = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    if cuda:
        net = net.cuda()
    trainDataNumber = 10000
    # 采用的是 matconvnet 的配置
    batch_size = 100
    maxEpoch = 20
    lr = 0.001
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
                print("iter {} loss: {}".format(index, loss))
        print("time cost of epoch {}: {}".format(epoch, time.time() - time1))

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
    plt.plot([i for i in range(len(losses))], losses)