from torchvision.datasets import EMNIST
import numpy as np
from PIL import Image


class iEMNIST(EMNIST):
    def __init__(self, root,
                 split='byclass',  # EMNIST has multiple splits
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super(iEMNIST, self).__init__(root,
                                      split=split,
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)

        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTrainData(self, classes, exemplar_set, exemplar_label_set):
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        for label in classes:
            # Convert tensor data to numpy if needed
            if hasattr(self.data, 'numpy'):
                data_array = self.data.numpy()
            else:
                data_array = self.data
                
            if hasattr(self.targets, 'numpy'):
                targets_array = self.targets.numpy()
            else:
                targets_array = self.targets
                
            data = data_array[np.array(targets_array) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                if hasattr(self.data, 'numpy'):
                    data_array = self.data.numpy()
                else:
                    data_array = self.data
                    
                if hasattr(self.targets, 'numpy'):
                    targets_array = self.targets.numpy()
                else:
                    targets_array = self.targets
                    
                data = data_array[np.array(targets_array) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainItem(self, index):
        # EMNIST data is grayscale, so we need to handle it properly
        img_data = self.TrainData[index]
        
        # If data is already 2D (H, W), convert to PIL Image directly
        if len(img_data.shape) == 2:
            img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        else:
            # If data is 1D, reshape it (assuming 28x28 for EMNIST)
            img = Image.fromarray(img_data.reshape(28, 28).astype(np.uint8), mode='L')
        
        target = self.TrainLabels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return index, img, target

    def getTestItem(self, index):
        img_data = self.TestData[index]
        
        # If data is already 2D (H, W), convert to PIL Image directly
        if len(img_data.shape) == 2:
            img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        else:
            # If data is 1D, reshape it (assuming 28x28 for EMNIST)
            img = Image.fromarray(img_data.reshape(28, 28).astype(np.uint8), mode='L')
            
        target = self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        if self.target_test_transform:
            target = self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData) > 0:
            return self.getTrainItem(index)
        elif len(self.TestData) > 0:
            return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) > 0:
            return len(self.TrainData)
        elif len(self.TestData) > 0:
            return len(self.TestData)
        else:
            return 0

    def get_image_class(self, label):
        if hasattr(self.data, 'numpy'):
            data_array = self.data.numpy()
        else:
            data_array = self.data
            
        if hasattr(self.targets, 'numpy'):
            targets_array = self.targets.numpy()
        else:
            targets_array = self.targets
            
        return data_array[np.array(targets_array) == label]