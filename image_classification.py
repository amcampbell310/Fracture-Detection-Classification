import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Constants
IMAGE_SIZE = (373, 454)  # Corrected image size to match model input shape
NUM_FRAC_IMAGES = 612
NUM_NON_FRAC_IMAGES = 620
DATASET_DIR = ''
CSV_FILE = ''

# Load dataset from CSV
df = pd.read_csv(CSV_FILE)

# Sample the desired number of fractured and non-fractured images
fractured_images = df[df['fractured'] == 1]['image_id'].sample(NUM_FRAC_IMAGES, random_state=42)
non_fractured_images = df[df['fractured'] == 0]['image_id'].sample(NUM_NON_FRAC_IMAGES, random_state=42)

# Load and preprocess images
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, IMAGE_SIZE[::-1])  # Corrected resizing
    img = img / 255.0  # Normalize pixel values
    return img

X = []
y = []

# Load fractured images
for image_name in fractured_images:
    image_path = os.path.join(DATASET_DIR, 'Fractured', image_name)
    X.append(load_and_preprocess_image(image_path))
    y.append(1)  # Fractured

# Load non-fractured images
for image_name in non_fractured_images:
    image_path = os.path.join(DATASET_DIR, 'Non_fractured', image_name)
    X.append(load_and_preprocess_image(image_path))
    y.append(0)  # Non-fractured

X = np.array(X)
y = np.array(y)

# Split dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Save the model
model.save("")


