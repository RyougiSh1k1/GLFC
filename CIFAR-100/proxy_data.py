import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd

class Proxy_Data():
    def __init__(self, test_transform=None):
        super(Proxy_Data, self).__init__()
        self.test_transform = test_transform
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, new_set, new_set_label):
        datas, labels = [], []
        self.TestData, self.TestLabels = [], []
        if len(new_set) != 0 and len(new_set_label) != 0:
            datas = [exemplar for exemplar in new_set]
            for i in range(len(new_set)):
                length = len(datas[i])
                labels.append(np.full((length), new_set_label[i]))

        if len(datas) > 0:  # Only concatenate if there's data
            self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        return img, target

    def __getitem__(self, index):
        if len(self.TestData) > 0:
            return self.getTestItem(index)

    def __len__(self):
        # Fixed: Check if TestData is a numpy array or list
        if isinstance(self.TestData, np.ndarray):
            return self.TestData.shape[0] if self.TestData.size > 0 else 0
        elif isinstance(self.TestData, list):
            return len(self.TestData)
        else:
            return 0