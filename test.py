
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from PIL import Image

# define the same CNN model structure
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(32 * 4 * 4, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

#current model's training results:: average loss: 0.0229, accuracy: 99.04%
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()


custom_img_dir = "C:\\Users\\ICT\\Machine Learning\\myDigits"

# Define the same format as MNIST
custom_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


loss_fn = nn.CrossEntropyLoss()


total_loss = 0.0
correct = 0
total = 0


for filename in sorted(os.listdir(custom_img_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(custom_img_dir, filename)

        try:
            true_label = int(filename[0])
        except ValueError:
            print(f"Skipping {filename} – could not extract label.")
            continue

        img = Image.open(img_path)
        input_tensor = custom_transform(img).unsqueeze(0).to(device)
        target_tensor = torch.tensor([true_label], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
            loss = loss_fn(output, target_tensor)

        total_loss += loss.item()
        correct += (pred == true_label)
        total += 1

        print(f"{filename} → Predicted: {pred}, True: {true_label}")
        plt.imshow(input_tensor.squeeze().cpu(), cmap='gray')
        plt.title(f"{filename} → Pred: {pred} / True: {true_label}")
        plt.axis('off')
        plt.show()


if total > 0:
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    print(f"\nCustom Test Set Accuracy: {accuracy:.2f}%")
    print(f"Average Test Loss: {avg_loss:.4f}")
else:
    print("No valid test images found.")