import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataSet(Dataset):
    def __init__(self, images, labels, number=-1):
        assert images.shape[0] == labels.shape[0]
        if number >= 0:
            self.images = images[:number, :, :]
            self.labels = labels[:number]
        else:
            self.images = images
            self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

