from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch

class MnistData:
    
    def __init__(self, batch_size: int):
        transform = transforms.ToTensor()
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self._batch_size = batch_size

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        flat_train_data = self._flatten_dataset(self.train_data)
        flat_test_data = self._flatten_dataset(self.test_data)

        train_loader = DataLoader(flat_train_data, batch_size=self._batch_size, shuffle=True)
        test_loader = DataLoader(flat_test_data, batch_size=self._batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def visualize_item(self, index: int, is_train: bool = True):  
        image, label = self.train_data[index] if is_train else self.test_data[index]
        plt.figure(figsize=(6, 6))
        plt.imshow(image.squeeze(), cmap='gray')  # .squeeze() removes the channel dimension
        plt.title(f'Label: {label}')
        plt.colorbar()
        plt.show()
    
    def _flatten_dataset(self, dataset: TensorDataset) -> TensorDataset:
        images = []
        labels = []
        for img, label in dataset:
            images.append(img.view(-1))
            labels.append(label)
        
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return TensorDataset(images, labels)
    