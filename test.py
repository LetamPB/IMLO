import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


# set up device
device = torch.device("cpu")

# same training transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 64

# load dataset
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# same neural network module in train file


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # dataset images are RGB and thus have 3 input channels
        # padding added to avoid misisng edge of image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # pooling layer helps wiht downsampling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # fully connected
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # conv1 -> ReLU -> pool
        x = self.pool(torch.relu(self.conv2(x)))  # conv2 -> ReLU -> pool
        x = torch.flatten(x, 1)               # flatten for linear layer
        x = torch.relu(self.fc1(x))               # fc1 -> ReLU
        x = torch.relu(self.fc2(x))               # fc2 -> ReLU
        x = self.fc3(x)                       # final output layer
        return x


# loading model
model = CNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

with torch.no_grad():
    for input_images, labels in test_loader:
        input_images, labels = input_images.to(device), labels.to(device)
        outputs = model(input_images)
        _, predicted = torch.max(outputs.data, 1)  # Get prediction
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test images: {accuracy:.2f}%")
