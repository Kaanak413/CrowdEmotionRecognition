import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import layers, models, applications, regularizers
from datetime import datetime  # Import for timestamp
from keras import Optimizer
from keras import callbacks
# Path to your dataset
path = '/home/kaan/Desktop/Projects-ML/EmotionRecognition/super/emotion_images/'
dataset_path = os.listdir(path)

# Print the classes in the dataset
print("Types of classes labels found: ", dataset_path)

# Define constants
im_size = 600
batch_size = 4
NUM_CLASSES = len(dataset_path)  # Assume you have 6 classes

img_size = (600, 600)

# # Split dataset paths into training and validation sets (80% train, 20% validation)
# train_paths, val_paths = train_test_split(dataset_path, test_size=0.2, random_state=42)

# # Function to generate batches of images and labels
# def image_data_generator(dataset_paths, batch_size, im_size):
#     label_encoder = LabelEncoder()
#     label_encoder.fit(dataset_paths)  # Encode labels based on folder names
    
#     while True:  # Infinite loop for the generator
#         images = []
#         labels = []
        
#         for i in dataset_paths:  # For each class directory
#             data_path = path + str(i)
#             filenames = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

#             for f in filenames:
#                 img = cv2.imread(data_path + '/' + f)
#                 if img is not None:
#                     img = cv2.resize(img, (im_size, im_size))
#                     images.append(img)
#                     labels.append(i)
                    
#                 # If batch size is reached, yield the batch and reset
#                 if len(images) == batch_size:
#                     images = np.array(images).astype('float32') / 255.0
#                     labels = np.array(labels)
#                     labels = label_encoder.transform(labels)  # Encode labels
#                     labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)  # One-hot encode
                    
#                     # Yield the batch and clear the lists for the next batch
#                     yield images, labels
#                     images = []
#                     labels = []

#         # Yield any remaining images in the final batch
#         if len(images) > 0:
#             images = np.array(images).astype('float32') / 255.0
#             labels = np.array(labels)
#             labels = label_encoder.transform(labels)  # Encode labels
#             labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)  # One-hot encode
#             yield images, labels

# # Define the generator for training and validation data
# train_gen = image_data_generator(train_paths, batch_size, im_size)
# val_gen = image_data_generator(val_paths, batch_size, im_size)

# # Define steps per epoch and validation steps
# total_train_images = sum([len(os.listdir(path + d)) for d in train_paths if os.path.isdir(path + d)])
# total_val_images = sum([len(os.listdir(path + d)) for d in val_paths if os.path.isdir(path + d)])

# steps_per_epoch = total_train_images // batch_size
# validation_steps = total_val_images // batch_size

# Define the model

train_dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=img_size,  # Resize to 600x600
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    verbose=True
)

# Validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=img_size,  # Resize to 600x600
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    verbose=True
)

initial_learning_rate = 0.001  # Starting learning rate
# optimizer = Optimizer.Adam(learning_rate=initial_learning_rate)
def create_custom_efficientnet(im_size=600, num_classes=NUM_CLASSES, base_dropout_rate=0.5, l2_reg=0.001):
    # Define input layer
    inputs = layers.Input(shape=(im_size, im_size, 3))
    
    # Load EfficientNetB7 without the top classification layer, but keep the global average pooling
    base_model = applications.EfficientNetB7(include_top=False, weights=None, input_tensor=inputs)
    
    # Global average pooling to flatten features
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    # Add a dense layer with L2 regularization and dropout
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)  # Add Batch Normalization after the dense layer
    x = layers.Dropout(base_dropout_rate)(x)  # Add dropout

    # Output layer with softmax activation for classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs, outputs)
    
    return model

def create_custom_resnet50(im_size=224, num_classes=NUM_CLASSES, base_dropout_rate=0.5, l2_reg=0.001):
    # Define input layer
    inputs = layers.Input(shape=(im_size, im_size, 3))
    
    # Load ResNet50 without the top classification layer, but keep the global average pooling
    base_model = applications.ResNet50(include_top=False, weights=None, input_tensor=inputs)
    
    # Global average pooling to flatten features
    x = layers.GlobalAveragePooling2D()(base_model.output)
    
    # Add a dense layer with L2 regularization and dropout
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)  # Add Batch Normalization after the dense layer
    x = layers.Dropout(base_dropout_rate)(x)  # Add dropout

    # Output layer with softmax activation for classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs, outputs)
    
    return model
im_size = 600  # Adjust based on your input image size
model = create_custom_efficientnet(im_size=im_size, num_classes=NUM_CLASSES)
model = create_custom_resnet50(im_size=im_size, num_classes=NUM_CLASSES)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model with training and validation data
with tf.device('/GPU:0'):  # Ensure GPU is used
    hist = model.fit(
        train_dataset,
        epochs=200,
        validation_data=val_dataset,
    )

# Plot training history
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp

def plot_hist(hist):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Model Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Model Loss")

    model_filename = f'emotion_recognition_model_{timestamp}.png' 
    plt.savefig(model_filename) 

    

plot_hist(hist)


model_filename = f'emotion_recognition_model_{timestamp}.h5'  # Filename with timestamp
model.save(model_filename)
print(f"Model saved to {model_filename}")
