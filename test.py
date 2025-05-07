import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import torch
import numpy as np

# Load MNIST dataset
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)

# Class distribution
labels = [label for _, label in mnist_data]
class_counts = np.bincount(labels)

# Plot: Class Distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=np.arange(10), y=class_counts, palette="muted")
plt.title("Digit Class Distribution in MNIST")
plt.xlabel("Digit")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# Plot: Average digit image (using first 1000 images)
image_stack = torch.stack([mnist_data[i][0] for i in range(1000)])
average_image = image_stack.mean(dim=0).squeeze()

plt.figure(figsize=(5, 5))
plt.imshow(average_image, cmap="gray")
plt.title("Average Digit Image (First 1000 Samples)")
plt.axis("off")
plt.tight_layout()
plt.show()
