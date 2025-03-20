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
    transforms.RandomAffine(0, shear=15, scale=(0.7, 1.3)),     # Distort images slightly
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),  # Simulate perspective shifts
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))                  # Normalize MNIST data
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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
    transform=test_transform,
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

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(16 * 4 * 4, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=4e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5) 
loss_fn = nn.CrossEntropyLoss()

# start training the model
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
    print(f"\nTEST SET:: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.2f}%)\n")

    return test_loss

# start training
patience = 3
counter = 0
best_loss = float('Inf')

for epoch in range(1, 11):
    train(epoch)
    test_loss = test()

    # adding early stops
    # checking for best test loss improvement
    # if test_loss < best_loss:
    #     best_loss = test_loss
    #     counter = 0

    #     torch.save(model.state_dict(), "mnist_cnn.pth")
    #     print(f"Model improved and saved at epoch {epoch} successfully!")
    # else:
    #     counter += 1
    #     print(f"No improvement for {counter}/{patience} epochs.")
    
    # # stops training if patience is reached
    # if counter >= patience:
    #     print("Early stopping triggered. Stopping training...")
    #     break

    # use if the model seems to overfit
    # scheduler.step()