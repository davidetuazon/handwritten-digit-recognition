from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch

# add noise to images during training
train_transform = transforms.Compose([
    transforms.RandomRotation(10),                              # Randomly rotate images by ±10 degrees
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),     # Distort images slightly
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Simulate perspective shifts
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))                  # Normalize MNIST data
])

# download data to be used for the model
train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=train_transform,
    download=True
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

# load data into loaders
loaders = {
    'train': DataLoader(train_data,
                        batch_size=64,
                        shuffle=True,
                        num_workers=0),

    'test': DataLoader(test_data,
                        batch_size=128,
                        shuffle=True,
                        num_workers=0)
}

# define model structure
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

# training of the model
def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")

# testing of the model
def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f"\nTEST SET:: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%)\n")

# start training
for epoch in range(1, 11):
    train(epoch)
    test()

torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved successfully!")