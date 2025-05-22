import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split

# GPU is not allowed so we select CPU
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 64

# load CIFAR-10 dataset
training_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

# splitting dataset between training(45k) and validation(5k)
train_size = 45000
val_size = 5000
train_dataset, val_dataset = random_split(training_set, [train_size, val_size])

# loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)


# define neural network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

    def forward(self, x):
        pass
