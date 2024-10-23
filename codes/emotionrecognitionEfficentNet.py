import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import layers, models, applications
from datetime import datetime  # Import for timestamp

# Path to your dataset
path = 'super/emotion_images/'
dataset_path = os.listdir(path)

# Print the classes in the dataset
print("Types of classes labels found: ", dataset_path)

# Define constants
im_size = 600
batch_size = 4
NUM_CLASSES = len(dataset_path)  # Assume you have 6 classes

# Split dataset paths into training and validation sets (80% train, 20% validation)
train_paths, val_paths = train_test_split(dataset_path, test_size=0.2, random_state=42)

# Function to generate batches of images and labels
def image_data_generator(dataset_paths, batch_size, im_size):
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset_paths)  # Encode labels based on folder names
    
    while True:  # Infinite loop for the generator
        images = []
        labels = []
        
        for i in dataset_paths:  # For each class directory
            data_path = path + str(i)
            filenames = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            for f in filenames:
                img = cv2.imread(data_path + '/' + f)
                if img is not None:
                    img = cv2.resize(img, (im_size, im_size))
                    images.append(img)
                    labels.append(i)
                    
                # If batch size is reached, yield the batch and reset
                if len(images) == batch_size:
                    images = np.array(images).astype('float32') / 255.0
                    labels = np.array(labels)
                    labels = label_encoder.transform(labels)  # Encode labels
                    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)  # One-hot encode
                    
                    # Yield the batch and clear the lists for the next batch
                    yield images, labels
                    images = []
                    labels = []

        # Yield any remaining images in the final batch
        if len(images) > 0:
            images = np.array(images).astype('float32') / 255.0
            labels = np.array(labels)
            labels = label_encoder.transform(labels)  # Encode labels
            labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)  # One-hot encode
            yield images, labels

# Define the generator for training and validation data
train_gen = image_data_generator(train_paths, batch_size, im_size)
val_gen = image_data_generator(val_paths, batch_size, im_size)

# Define steps per epoch and validation steps
total_train_images = sum([len(os.listdir(path + d)) for d in train_paths if os.path.isdir(path + d)])
total_val_images = sum([len(os.listdir(path + d)) for d in val_paths if os.path.isdir(path + d)])

steps_per_epoch = total_train_images // batch_size
validation_steps = total_val_images // batch_size

# Define the model
inputs = layers.Input(shape=(im_size, im_size, 3))
outputs = applications.EfficientNetB7(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
model = models.Model(inputs, outputs)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model with training and validation data
with tf.device('/GPU:0'):  # Ensure GPU is used
    hist = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        validation_data=val_gen,
        validation_steps=validation_steps
    )

# Plot training history

def plot_histAcc(hist):
    plt.plot(hist.history["accuracy"], label="train accuracy")
    plt.plot(hist.history["val_accuracy"], label="validation accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.savefig("training_accuracy.png")  # Save the plot to a file
    print("Plot saved to training_accuracy.png")

plot_histAcc(hist)

def plot_histLoss(hist):
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="validation loss")
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.savefig("training_loss.png")  # Save the plot to a file
    print("Plot saved to training_loss.png")

plot_histAcc(hist)
plot_histLoss(hist)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
model_filename = f'emotion_recognition_model_{timestamp}.h5'  # Filename with timestamp
model.save(model_filename)
print(f"Model saved to {model_filename}")
