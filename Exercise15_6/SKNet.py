import torch
from torch import nn
from DataReader import DataReader
from MNISTDataSet import MNISTDataSet
from torch.utils.data import DataLoader
import time
import os

class SKNet(nn.Module):
    def __init__(self, cAlpha=1):
        super(SKNet, self).__init__()
        """
        SKNet 的做法:
            PS: 就是把前面 BASENET 的第一个 conv 拆成了两个 conv
                同时 每个 conv 后面都加一个 BN 和 ReLU
            1. 3 * 3 * 1 * 20 conv
            2. maxpooling
            3. 3 * 3 * 20 * 20 conv
            4. maxpooling
        """
        self.conv1_1 = nn.Conv2d(1, int(20*cAlpha), 3, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(int(20*cAlpha))
        self.relu1_1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(int(20*cAlpha), int(20*cAlpha), 3, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(int(20*cAlpha))
        self.relu1_2 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(int(20*cAlpha), int(50*cAlpha), 5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(int(50*cAlpha))
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(int(50*cAlpha), int(500*cAlpha), 4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(int(500*cAlpha))
        self.relu3 = nn.ReLU(True)
        self.conv4 = nn.Conv2d(int(500*cAlpha), 10, 1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(10)
        self.relu4 = nn.ReLU(True)
        self.dropout = nn.Dropout(0.7)
        self.softmax = nn.Softmax(dim=1)
        self.pipeline = nn.Sequential(
            self.conv1_1,
            self.bn1_1,
            self.relu1_1,
            self.conv1_2,
            self.bn1_2,
            self.relu1_2,
            self.maxpool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.maxpool2,
            self.conv3,
            self.bn3,
            self.relu3,
            self.conv4,
            self.bn4,
            self.relu4
        )
        # 参数初始化
        self.paramInit()
        # PS: CrossEntropyLoss 的 target 的每个 entry 中期望的是一个 long 类型的 长度为 1 的数组
        self.lossFunction = nn.CrossEntropyLoss()
    # 参数初始化
    def paramInit(self):
        moduleL = []
        moduleL.append(self.conv1_1)
        moduleL.append(self.conv1_2)
        moduleL.append(self.conv2)
        moduleL.append(self.conv3)
        moduleL.append(self.conv4)
        for module in moduleL:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, hasDropout=False):
        assert x.shape[1:] == (1, 28, 28)
        x = self.pipeline(x)
        x = x.view(x.shape[0], -1)
        if hasDropout:
            x = self.dropout(x)
        x = self.softmax(x)
        return x

    def loss(self, result, GTResult):
        assert result.shape[1] == 10
        return self.lossFunction(result, GTResult)


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
    lr = 0.01
    # PS: 很神奇的是用 Adam 的时候损失函数不收敛... 一点都不变化... 用 Adam 的时候降低学习率就好了...
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # print(trainImages.shape, trainLabels.shape, testImages.shape, testLabels.shape)
    trainDataset = MNISTDataSet(trainImages, trainLabels, trainDataNumber)
    testDataset = MNISTDataSet(testImages, testLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=32)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=True, num_workers=32)
    index = 0
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
            optimizer.step()
            if index % 100 == 0:
                print("iter {} loss: {}".format(index, loss))
        print("time cost of epoch {}: {}".format(epoch, time.time() - time1))


    count = 0
    correct = 0
    net = net.eval()
    for i, data in enumerate(testLoader):
        count += 1
        if count%1000 == 0:
            print("current: {}/{}".format(correct,  count))
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
    print("lr: {} accuracy rate: {}".format(lr, correct/count))



