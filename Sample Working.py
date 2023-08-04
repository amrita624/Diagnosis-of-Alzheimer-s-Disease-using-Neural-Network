import torch
import torch.nn as nn
from PIL import Image
import numpy as np

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

# Load the trained model
model_path = 'alzheimer_net.pth'
model_state_dict = torch.load(model_path)
model = AlzheimerNet()
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

# Define a function to predict the class of a single image
def predict_class(image):
    # Convert the image to a PyTorch tensor
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    
    # Use the trained model to make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = nn.functional.softmax(output, dim=1)
        class_index = torch.argmax(probabilities, dim=1)
        class_index = class_index.item()
        return class_index

# Load an example image
image_path = "C:/Users/KIIT/OneDrive/Desktop/Alzheimer_s Dataset/test/VeryMildDemented/29 (18).jpg"
image = Image.open(image_path)
image = image.resize((224, 224))
image = np.array(image)

# Use the trained model to predict the class of the image
predicted_class = predict_class(image)
print('Predicted class:', predicted_class)