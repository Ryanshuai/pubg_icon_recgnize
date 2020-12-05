import os
import cv2
import random
import torch
from torch.utils import data
from PIL import Image


class IconDataset(data.Dataset):
    def __init__(self, root, length, transform):
        self.dataRoot = root
        self.kinds = os.listdir(root)
        self.n = length
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        kind_i = random.randint(0, len(self.kinds) - 1)
        kind = self.kinds[kind_i]

        kindFolder = os.path.join(self.dataRoot, kind)
        kindImNames = os.listdir(kindFolder)
        kindImPaths = [os.path.join(kindFolder, name) for name in kindImNames]
        index %= len(kindImPaths)
        kindImPath = kindImPaths[index]

        img = cv2.imread(kindImPath)
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, kind_i
