import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Set the path to your dataset
cat_dir = r'C:\Users\sanik\OneDrive\Desktop\Machine Learning\Task 3\cats' #cats images
dog_dir = r'C:\Users\sanik\OneDrive\Desktop\Machine Learning\Task 3\dogs' #dogs images

# Image processing settings
image_size = (64, 64)  # Resize all images to 64x64
categories = ['cat', 'dog']
label_map = {'cat': 0, 'dog': 1}  # Labels for cats and dogs

# Load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert image to grayscale
            img = img.resize(image_size)  # Resize image
            img_array = np.array(img).flatten()  # Flatten the image into a 1D array
            images.append(img_array)
            labels.append(label)
    return images, labels

# Load data for both categories
cat_images, cat_labels = load_images_from_folder(cat_dir, label_map['cat'])
dog_images, dog_labels = load_images_from_folder(dog_dir, label_map['dog'])

# Combine images and labels
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (SVM performs better with normalized data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)  # You can try other kernels like 'rbf'
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Function to preprocess and classify a new image
def classify_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize(image_size)  # Resize image to the same size used for training
    img_array = np.array(img).flatten()  # Flatten the image into a 1D array
    img_array = scaler.transform([img_array])  # Scale the image as per training data
    prediction = svm_model.predict(img_array)  # Predict using the trained model
    label = categories[prediction[0]]  # Convert numerical label back to category
    return label

# Test the classify_image function
test_image_path = r'C:\Users\sanik\OneDrive\Desktop\test\5.jpg'
result = classify_image(test_image_path)
print(f"The image at {test_image_path} is classified as a {result}.")
