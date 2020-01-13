import numpy as np
import struct

# MNIST 数据文件的路径
defaultTrainImagesPath = "./MNIST/train-images-idx3-ubyte"
defaultTrainLabelsPath = "./MNIST/train-labels-idx1-ubyte"
defaultTestImagesPath = "./MNIST/t10k-images-idx3-ubyte"
defaultTestLabelsPath = "./MNIST/t10k-labels-idx1-ubyte"

# 进行数据读取的信息, 实际上四个文件对应着两种数据格式,
# 一种是存储 Images 的(idx3), 一种是存储 Labels 的(idx1)
class DataReader(object):
    def __init__(self,
                 trainImagesPath=defaultTrainImagesPath,
                 trainLabelsPath=defaultTrainLabelsPath,
                 testImagesPath=defaultTestImagesPath,
                 testLabelsPath=defaultTestLabelsPath):
        self.trainImagesPath = trainImagesPath
        self.trainLabelsPath = trainLabelsPath
        self.testImagesPath = testImagesPath
        self.testLabelsPath = testLabelsPath

    # 加载 训练图片 数据
    def loadTrainImages(self):
        assert self.trainImagesPath is not None
        return self.loadIdx3(self.trainImagesPath)

    # 加载 训练Label 数据
    def loadTrainLabels(self):
        assert self.trainLabelsPath is not None
        return self.loadIdx1(self.trainLabelsPath)

    # 加载 测试图片 数据
    def loadTestImages(self):
        assert self.testImagesPath is not None
        return self.loadIdx3(self.testImagesPath)

    # 加载 测试Label 数据
    def loadTestLables(self):
        assert self.testLabelsPath is not None
        return self.loadIdx1(self.testLabelsPath)

    # 从 idx3 类型的文件中进行数据的读取
    def loadIdx3(self, filePath):
        """
        idx3 格式:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        """
        byteData = open(filePath, 'rb').read()
        # 进行文件头的解析, 按照上面描述的 idx3 的格式来说, 分别对应的是
        #   (1). magic number (2). images 数量 (3). 行数 (4). 列数
        offset = 0
        # > 符号表示 大端, 按照原字节数, < 表示 小端, 按照原字节数, @ （默认) 表示使用本机的字符顺序, 并且凑够4字节, = 表示 按照本机字符顺序, 但是按照原字节数,  ! 表示 大端, 按照原字节数
        # 格式符号中 c: data(1) x: 填充字节 b: signed char(1) B: unsigned char(1) ?: bool(1) h: short(2) H: unsigned short(2) i: int(4) I: unsigned int(4)
        #          l: long(4) L: unsigned long(4) q: long long(8) Q: unsigned long long(8) f: float(4) d: double(8) s: char[]
        fmt_header = ">iiii" # 读取前面四个数字
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, byteData, offset)
        print("{} messages: magic_number: {}, num_images: {}, num_rows: {}, num_cols: {}".format(filePath, magic_number, num_images, num_rows, num_cols))

        # 开始对剩下的数据进行解析
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = ">{}B".format(str(image_size))
        # 先建立一个空的 array 数组, 后面读出来之后直接填充进去就好了
        images = np.empty((num_images, num_rows, num_cols))
        for i in range(num_images):
            if ( i + 1 ) % 10000 == 0:
                print("{} 中已经解析了 {} 张图片".format(filePath, i+1))
            images[i] = np.array(struct.unpack_from(fmt_image, byteData, offset)).reshape((num_rows, num_cols))
            offset += struct.calcsize(fmt_image)

        return images

    # 从 idx1 类型的文件中进行数据的读取
    def loadIdx1(self, filePath):
        """
        idx1 格式:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.
        """
        byteData = open(filePath, 'rb').read()
        offset = 0
        fmt_header = ">ii"
        magic_number, num_images = struct.unpack_from(fmt_header, byteData, offset)
        print("{} messages: magic_number: {}, num_labels: {}".format(filePath, magic_number, num_images))

        offset += struct.calcsize(fmt_header)
        fmt_image = ">B"
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print("{} 中已经解析了 {} 张图片".format(filePath, i + 1))
            labels[i] = struct.unpack_from(fmt_image, byteData, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels