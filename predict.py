
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# define the same CNN model structure
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(16 * 4 * 4, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc2(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# Load EMNIST/MNIST test dataset

# test_data = datasets.MNIST(
#     root='data',
#     train=False,
#     transform=ToTensor(),
#     download=True
# )

test_data = datasets.EMNIST(
    root='data',
    split='digits',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, -1, [1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, [2])),
        transforms.Normalize((0.1307,), (0.3081,))  
    ]),
    download=True
)

test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

data_iter = iter(test_loader)
data, target = next(data_iter)
data = data.to(device)

# Run predictions for each image
for i in range(10):
    image = data[i].unsqueeze(0)
    output = model(image)
    prediction = output.argmax(dim=1, keepdim=True).item()

    print(f"Predicted Label: {prediction}")
    image = image.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.show()

# Define loss function
loss_fn = nn.CrossEntropyLoss()

test_loss = 0
correct = 0
total_samples = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += loss_fn(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += target.size(0)

test_loss /= len(test_loader)
accuracy = 100. * correct / total_samples

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")