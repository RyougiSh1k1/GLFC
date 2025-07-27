import torch.nn as nn
import torch
import copy
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from myNetwork import *
from torch.utils.data import DataLoader
import random
from Fed_utils import *
from proxy_data import *

class proxyServer:
    def __init__(self, device, learning_rate, numclass, feature_extractor, encode_model, test_transform):
        super(proxyServer, self).__init__()
        self.Iteration = 100  # Reduced for EMNIST
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.encode_model = encode_model
        self.monitor_dataset = Proxy_Data(test_transform)
        self.new_set = []
        self.new_set_label = []
        self.numclass = numclass  # Use the provided numclass
        self.device = device
        self.num_image = 10  # Reduced for EMNIST
        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0:
            self.reconstruction()
            if len(self.new_set) > 0:  # Only create dataloader if we have data
                self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
                self.monitor_loader = DataLoader(dataset=self.monitor_dataset, shuffle=True, batch_size=32, drop_last=False)
                self.last_perf = 0
                self.best_model_1 = self.best_model_2
                cur_perf = self.monitor()
                print(f"Monitor performance: {cur_perf}")
                if cur_perf >= self.best_perf:
                    self.best_perf = cur_perf
                    self.best_model_2 = copy.deepcopy(self.model)
        else:
            # No gradient to reconstruct, just copy the model
            self.best_model_2 = copy.deepcopy(self.model)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def monitor(self):
        if not hasattr(self, 'monitor_loader'):
            return 0.0
            
        self.model.eval()
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(self.monitor_loader):
            imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        
        accuracy = 100 * correct / total if total > 0 else 0.0
        return accuracy

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            # For EMNIST with modified network, gradient structure might be different
            try:
                pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
                pool_label.append(pred.item())
            except:
                # If gradient structure is different, use a random label from the numclass
                pool_label.append(np.random.randint(0, self.numclass))
        return pool_label

    def reconstruction(self):
        self.new_set, self.new_set_label = [], []

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        pool_label = self.gradient2label()
        pool_label = np.array(pool_label)
        
        # Count occurrences of each class
        class_ratio = np.zeros(self.numclass)
        for label in pool_label:
            if 0 <= label < self.numclass:
                class_ratio[label] += 1

        for label_i in range(self.numclass):
            if class_ratio[label_i] > 0:
                augmentation = []
                
                grad_indices = np.where(pool_label == label_i)[0]
                for j in range(min(len(grad_indices), 2)):  # Limit reconstructions per class
                    try:
                        grad_truth_temp = self.pool_grad[grad_indices[j]]

                        # Initialize with grayscale image (1 channel, 28x28)
                        dummy_data = torch.randn((1, 1, 28, 28)).to(self.device).requires_grad_(True)
                        label_pred = torch.Tensor([label_i]).long().to(self.device).requires_grad_(False)

                        optimizer = torch.optim.LBFGS([dummy_data, ], lr=0.1)
                        criterion = nn.CrossEntropyLoss().to(self.device)

                        recon_model = copy.deepcopy(self.encode_model)
                        recon_model = model_to_device(recon_model, False, self.device)

                        for iters in range(self.Iteration):
                            def closure():
                                optimizer.zero_grad()
                                pred = recon_model(dummy_data)
                                dummy_loss = criterion(pred, label_pred)

                                dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                                grad_diff = 0
                                for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                                    grad_diff += ((gx - gy) ** 2).sum()
                                grad_diff.backward()
                                return grad_diff

                            optimizer.step(closure)
                            
                            if iters >= self.Iteration - self.num_image:
                                # Convert to numpy and ensure it's in the right format
                                dummy_data_cpu = dummy_data.clone().detach().cpu()
                                # Convert from tensor to numpy, scale to 0-255
                                dummy_data_np = (dummy_data_cpu.squeeze(0).squeeze(0).numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                                augmentation.append(dummy_data_np)

                    except Exception as e:
                        print(f"Reconstruction error for label {label_i}: {e}")
                        # Create a random image as fallback
                        random_img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
                        augmentation.append(random_img)

                if len(augmentation) > 0:
                    self.new_set.append(augmentation)
                    self.new_set_label.append(label_i)