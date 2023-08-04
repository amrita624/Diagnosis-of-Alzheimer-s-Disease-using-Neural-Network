import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Define the dataset class
class AlzheimerDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')  
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')  
  
# Convert the labels to integers
label_to_int = {'NonDemented': 0, 'VeryMildDemented': 1, 'MildDemented': 2, 
                'ModerateDemented': 3}
test_labels = np.array([label_to_int[label] for label in test_labels])
train_labels = np.array([label_to_int[label] for label in train_labels])

# Convert the data to PyTorch tensors
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)
train_images = torch.tensor(train_images, dtype=torch.float32)
trin_labels = torch.tensor(train_labels, dtype=torch.long)

# Create the dataloader
test_dataset = AlzheimerDataset(test_images, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_dataset = AlzheimerDataset(train_images, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

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

# Load the saved weights
net = AlzheimerNet()
net.load_state_dict(torch.load('alzheimer_net.pth'))

# Test the neural network
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test set: {:.2f}%'.format(100 * correct / total))

# Create a function to generate the confusion matrix
def get_confusion_matrix(net, dataloader):
    # Set the model to evaluation mode
    net.eval()
    # Create the confusion matrix
    num_classes = 4
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = net(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                confusion_matrix[labels[j], predicted[j]] += 1
    return confusion_matrix

# Load the saved model
net = AlzheimerNet()
net.load_state_dict(torch.load('alzheimer_net.pth'))

# Generate the confusion matrix for the training data
train_confusion_matrix = get_confusion_matrix(net, train_dataloader)
print('Training Confusion Matrix:')
print(train_confusion_matrix)

# Generate the confusion matrix for the test data
test_confusion_matrix = get_confusion_matrix(net, test_dataloader)
print('Test Confusion Matrix:')
print(test_confusion_matrix)


# Compute the predicted probabilities for the test set
y_probs = []
y_true = []
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images.unsqueeze(1))
        probs = nn.functional.softmax(outputs, dim=1)
        y_probs += probs.tolist()
        y_true += labels.tolist()

# Compute the ROC curve and AUC score
y_probs = np.array(y_probs)
y_true = np.array(y_true)
fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1], pos_label=1)
auc_score = roc_auc_score(y_true, y_probs[:, 1])

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc_score))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()