import torch
from torch import nn
from DataReader import DataReader
from MNISTDataSet import MNISTDataSet
from torch.utils.data import DataLoader
import time

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        """
        按照 Exercise 中题目的说明, 翻了半天翻到的 BASE 在这里:
            https://github.com/vlfeat/matconvnet/blob/master/examples/mnist/cnn_mnist_init.m
        整个网络的结构:
            PS: (1). 所有的 conv 都是 stride: 1 pad: 0 (2). 所有的 maxpooling 都是 stride 为 2
            1. 5 * 5 * 1 * 20 conv + 2 * 2 maxpooling (28 -> 24 -> 12)
            2. 5 * 5 * 20 * 50 conv + 2 * 2 maxpooling (12 -> 8 -> 4)
            3. 4 * 4 * 50 * 500 conv(实际上就是一个 FC 层) + ReLU
            4. classification layer 1 * 1 * 500 * 10 conv (其实就是一个 FC)
            5. 使用 softmaxLoss -> L = -Σy_j*log(s_j)
        """
        self.conv1 = nn.Conv2d(1, 20, 5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, 5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(50, 500, 4, stride=1, padding=0)
        self.ReLU = nn.ReLU(True)
        self.conv4 = nn.Conv2d(500, 10, 1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.pipeline = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.maxpool2,
            self.conv3,
            self.ReLU,
            self.conv4
        )
        # 参数初始化
        self.paramInit()
        # PS: CrossEntropyLoss 的 target 的每个 entry 中期望的是一个 long 类型的 长度为 1 的数组
        self.lossFunction = nn.CrossEntropyLoss()
    # 参数初始化
    def paramInit(self):
        moduleL = []
        moduleL.append(self.conv1)
        moduleL.append(self.conv2)
        moduleL.append(self.conv3)
        moduleL.append(self.conv4)
        for module in moduleL:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        assert x.shape[1:] == (1, 28, 28)
        x = self.pipeline(x)
        x = x.view(x.shape[0], -1)
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
    net = BaseNet()
    trainDataNumber = 10000
    # 采用的是 matconvnet 的配置
    batch_size = 100
    maxEpoch = 20
    lr = 0.02
    # PS: 很神奇的是用 Adam 的时候损失函数不收敛... 一点都不变化...
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-6)
    # print(trainImages.shape, trainLabels.shape, testImages.shape, testLabels.shape)
    trainDataset = MNISTDataSet(trainImages, trainLabels, trainDataNumber)
    testDataset = MNISTDataSet(testImages, testLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)
    index = 0
    for epoch in range(maxEpoch):
        time1 = time.time()
        for i, data in enumerate(trainLoader):
            index += 1
            images, label = data
            # print(images.shape, label.shape)
            images = images.unsqueeze(1)
            images = images.float()
            label = label.long()
            predict = net(images)
            loss = net.loss(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 100 == 0:
                print("iter {} loss: {}".format(index, loss))
        print("time cost of epoch {}: {}".format(epoch, time.time() - time1))


    count = 0
    correct = 0
    for i, data in enumerate(testLoader):
        count += 1
        if count%1000 == 0:
            print("current: {}/{}".format(correct,  count))
        image, label = data
        image = image.float()
        image = image.unsqueeze(1)
        label = label.long()
        predict = net(image)
        maxValue, maxIndex = torch.max(predict, 1)
        maxIndex = maxIndex.long()
        if maxIndex == label:
            correct += 1
    print("accuracy rate: {}".format(correct/count))



