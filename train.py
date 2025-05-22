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
# to have consistent split
generator = torch.Generator().manual_seed(42)
# splitting dataset between training(45k) and validation(5k)
train_size = 45000
val_size = 5000
train_dataset, val_dataset = random_split(
    training_set, [train_size, val_size], generator=generator)

# loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)


# define neural network model
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


# Loss function ,optimisation and the Model
model = CNN().to(device)
# loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 12

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()  # Set model to training mode

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()         # Clear gradients from previous step
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()               # Backpropagation
        optimizer.step()              # Update weights

        running_loss += loss.item()

        if (i + 1) % 100 == 0:  # Print every 100 mini-batches
            print(
                f"[Epoch {epoch+1}/{num_epochs}, Batch {i+1}] Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Training is finished")

torch.save(model.state_dict(), 'model.pth')
