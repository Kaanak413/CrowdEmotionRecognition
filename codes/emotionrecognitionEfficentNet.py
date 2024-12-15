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
from keras import optimizers
from keras import callbacks

# Path to your dataset
path = '/home/kaan/Desktop/Projects-ML/EmotionRecognition/super/emotion_images/'
dataset_path = os.listdir(path)

# Print the classes in the dataset
print("Types of classes labels found: ", dataset_path)

# Define constants
batch_size = 4
NUM_CLASSES = len(dataset_path)  # Assume you have 6 classes

img_size = (600, 600)


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
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal_and_vertical"),  # 50% chance of flipping horizontally and vertically
#     layers.RandomRotation(0.1)                    # Random rotation up to 20% of a full circle                        # Random zoom (up to 20%)
#                         # Random contrast adjustment (up to 20%)
# ])
# # def preprocess(image, label, augment=False):
# #     """
# #     Preprocess images by resizing, normalizing, and optionally augmenting.
# #     """
# #     image = tf.image.resize(image, img_size)  # Ensure size is consistent
# #     image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
# #     if augment:
# #         image = data_augmentation(image)  # Apply augmentations
# #     return image, label

# # train_dataset = train_dataset.map(lambda x, y: preprocess(x, y, augment=True),
# #                                   num_parallel_calls=tf.data.AUTOTUNE)

# # # Apply preprocessing without augmentation to the validation dataset
# # val_dataset = val_dataset.map(lambda x, y: preprocess(x, y, augment=False),
# #                               num_parallel_calls=tf.data.AUTOTUNE)

# # train_dataset = train_dataset.cache().shuffle(100).prefetch(buffer_size=4)
# # val_dataset = val_dataset.cache().prefetch(buffer_size=4)


initial_learning_rate = 0.005  # Starting learning rate
# optimizer = Optimizer.Adam(learning_rate=initial_learning_rate)
def create_custom_efficientnet(im_size=600, num_classes=NUM_CLASSES, base_dropout_rate=0.45, l2_reg=0.001):
    # Define input layer
    inputs = layers.Input(shape=(im_size, im_size, 3))
    
    # Load EfficientNetB7 without the top classification layer, but keep the global average pooling
    base_model = applications.EfficientNetB7(include_top=False, weights="imagenet", input_tensor=inputs)
    
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
def create_custom_resnet101(im_size=224, num_classes=NUM_CLASSES, base_dropout_rate=0.2, l2_reg=0.0003):
    # Define input layer
    inputs = layers.Input(shape=(im_size, im_size, 3))
    
    # Load ResNet101V2 with pretrained weights, excluding the top classification layer
    base_model = applications.ResNet101V2(include_top=False, weights="imagenet", input_tensor=inputs)
    
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
# model = create_custom_resnet50(im_size=im_size, num_classes=NUM_CLASSES)
#model = create_custom_resnet101(im_size=im_size, num_classes=NUM_CLASSES)

# Compile the model
epochSize=300

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',     
    patience=25,            
    restore_best_weights=True  
)
# decay_steps = epochSize
# alpha = 1e-6
# cosine_decay = optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps, alpha=alpha)
model.compile(optimizer=optimizers.Adam(learning_rate=initial_learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def lr_schedule(epoch, lr):
    if epoch < epochSize * 0.65:  # First 50% of epochs
        return lr  # Keep initial learning rate
    elif epoch < epochSize * 0.8:  # Next 30% of epochs
        return max(lr * 0.8, 1e-4)  # Reduce by 20%, with a higher minimum limit
    else:  # Last 20% of epochs
        return max(lr * 0.5, 1e-5)  # Reduce further, but not below 1e-6
 
# def lr_logger(epoch, learning_rate):
#     print(f"Epoch {epoch}: Learning Rate = {learning_rate}")
#     return learning_rate
# lr_callback = callbacks.LearningRateScheduler(lr_logger)

lrSchedule = callbacks.LearningRateScheduler(lr_schedule)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True
)
reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.25, patience=7, min_lr=1e-5, verbose=1
)
# Print model summary
model.summary()

# Train the model with training and validation data
with tf.device('/GPU:0'):  # Ensure GPU is used
    hist = model.fit(
        train_dataset,
        epochs=epochSize,
        validation_data=val_dataset,
        callbacks=[reduce_lr_on_plateau,lrSchedule]
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
