import cv2
import numpy as np
import os

# Define paths to the training and test dataset folders
train_path = 'C:/Users/KIIT/Downloads/alzheimers/Alzheimer_s Dataset/train'
test_path = 'C:/Users/KIIT/Downloads/alzheimers/Alzheimer_s Dataset/test'

# Define a function to preprocess images
def preprocess_image(img_path):
    # Read the image from file
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to a fixed size (e.g., 224x224 pixels)
    img = cv2.resize(img, (224, 224))
    # Normalize the pixel values to be between 0 and 1
    img = img / 255.0
    return img

# Define a function to preprocess a dataset folder
def preprocess_dataset(path):
    # Initialize empty lists to store the preprocessed images and their labels
    images = []
    labels = []
    # Loop over all the subfolders in the dataset folder
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        # Loop over all the image files in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg'):
                # Preprocess the image and append it to the list of images
                img_path = os.path.join(subfolder_path, filename)
                img = preprocess_image(img_path)
                images.append(img)
                # Append the label of the image to the list of labels
                labels.append(subfolder)
    # Convert the list of images and labels to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Preprocess the training and test datasets
train_images, train_labels = preprocess_dataset(train_path)
test_images, test_labels = preprocess_dataset(test_path)
# Preprocess the training and test datasets
train_images, train_labels = preprocess_dataset(train_path)
test_images, test_labels = preprocess_dataset(test_path)

# Save the preprocessed data to disk
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
