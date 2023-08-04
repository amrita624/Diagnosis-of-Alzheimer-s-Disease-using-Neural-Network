import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the dataset class
class AlzheimerDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')    

# Convert the labels to integers
label_to_int = {'NonDemented': 0, 'VeryMildDemented': 1, 'MildDemented': 2, 
                'ModerateDemented': 3}
train_labels = np.array([label_to_int[label] for label in train_labels])
test_labels = np.array([label_to_int[label] for label in test_labels])

# Convert the data to PyTorch tensors
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Create the dataloaders
train_dataset = AlzheimerDataset(train_images, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = AlzheimerDataset(test_images, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network architecture
class AlzheimerNet(nn.Module):
    def __init__(self):
        super(AlzheimerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 56 * 56)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# Create the neural network and the optimizer
net = AlzheimerNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the neural network
num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch {}/{} Batch {}/{} Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, len(train_dataloader), loss.item()))
            
# Save the model
torch.save(net.state_dict(), 'alzheimer_net.pth')