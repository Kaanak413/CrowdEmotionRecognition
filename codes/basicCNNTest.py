from keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models, applications
from keras import callbacks
import os
from PIL import Image
from pathlib import Path
import imghdr

path = '/home/kaan/Desktop/Projects-ML/EmotionRecognition/super/emotion_images/'
dataset_path = os.listdir(path)


# Print the classes in the dataset
print("Types of classes labels found: ", dataset_path)
im_size = 600
img_size = (600, 600)
batch_size = 16
NUM_CLASSES = len(dataset_path)  # Assume you have 6 classes

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


# Define the CNN model
def create_basic_cnn(input_shape=(600, 600, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Compile the model
cnn_model = create_basic_cnn()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model.summary()

# Early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
with tf.device('/GPU:0'):
    hist = cnn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50
    )

# Plot training history
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

    plt.savefig("training_with_basic_CNN.png") 

    

plot_hist(hist)
