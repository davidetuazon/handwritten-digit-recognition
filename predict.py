
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

# define the same CNN model structure
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# Load MNIST test dataset
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True)

# select test images (batch of 10)
for i in range(1, 11):
    data, target = test_data[i]
    data = data.unsqueeze(0).to(device)

    # Run prediction
    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True).item()

    # Print and visualize
    print(f"Predicted Label No.{i}: {prediction}")
    image = data.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.show()
