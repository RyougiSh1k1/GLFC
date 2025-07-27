import numpy as np
from PIL import Image
import torch

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
            con_label = np.concatenate((con_label,labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, new_set, new_set_label):
        datas, labels = [], []
        self.TestData, self.TestLabels = [], []
        if len(new_set) != 0 and len(new_set_label) != 0:
            datas = [exemplar for exemplar in new_set]
            for i in range(len(new_set)):
                length = len(datas[i])
                labels.append(np.full((length), new_set_label[i]))

        if len(datas) > 0:
            self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTestItem(self, index):
        # Handle grayscale images
        img_data = self.TestData[index]
        
        # Ensure the image is 2D
        if len(img_data.shape) == 3:
            img_data = img_data.squeeze()
        elif len(img_data.shape) == 1:
            img_data = img_data.reshape(28, 28)
            
        img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        target = self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        return img, target

    def __getitem__(self, index):
        if len(self.TestData) > 0:
            return self.getTestItem(index)
        else:
            # Return dummy data if no test data
            dummy_img = torch.zeros(1, 28, 28)
            return dummy_img, 0

    def __len__(self):
        if hasattr(self, 'TestData') and len(self.TestData) > 0:
            return self.TestData.shape[0]
        else:
            return 0