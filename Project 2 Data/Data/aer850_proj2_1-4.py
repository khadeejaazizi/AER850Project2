# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:04:39 2024

@author: Khadeeja Azizi
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")




#Step 1 - Data preprocessing

# Define image shape and batch size
IMG_WIDTH, IMG_HEIGHT = 500, 500
BATCH_SIZE = 32

# Set paths for data directories
train_dir = "./train"
valid_dir = "./valid"
test_dir = "./test"

# Data Augmentation for the training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True
)

# Only rescale validation data (no augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Create generators for train and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)




#Step 2 - Initial Neural Network Architecture

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout,Input

# Define the first CNN model 

# Initialize the model
model1 = Sequential()

#The First Convolutional layer
model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

# The MaxPooling layer
model1.add(MaxPooling2D(pool_size=(2, 2)))

# The Second Convolutional layer
model1.add(Conv2D(64, (3, 3), activation='relu'))

# MaxPooling layer
model1.add(MaxPooling2D(pool_size=(2, 2)))


# Flatten the output
model1.add(Flatten())

# Add Fully Connected layer
model1.add(Dense(128, activation='relu'))

# Add Dropout layer
model1.add(Dropout(0.5))

# The Output layer
model1.add(Dense(3, activation='softmax'))

# Compile the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model1.summary()

#number of trainable paramaters for this model is way too high for the dataset in this project






#Step 3 - Hyperparameter analysis

#increased number of convolutional layers from 2 to 5
#4 convolutional layers with progressively increasing filters (16, 32, 64, 128...). 
#This setup effectively captures both simple and more complex patterns in the images

#tweaked fully connected layer
#adjusted dropout
#got rid of some initial errors
#experimented with learning rate value


# Initialize the model
model2 = Sequential()

#input layer to define input shape
model2.add(Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

# First Convolutional layer
model2.add(Conv2D(16, (3, 3), activation='relu'))
# MaxPooling layer
model2.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional layer
model2.add(Conv2D(32, (3, 3), activation='relu'))
# MaxPooling layer
model2.add(MaxPooling2D(pool_size=(2, 2)))


# third Conv layer to increase depth and capture complex patterns
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

# fourth Conv layer to increase depth and capture complex patterns
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

#final conv layer
model2.add(Conv2D(256, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))


# Flatten the output
model2.add(Flatten())

# Fully Connected layer
model2.add(Dense(64, activation='relu'))

# Dropout layer (prevent overfitting)
model2.add(Dropout(0.5))

# Output layer (3 units for 3 classes)
model2.add(Dense(3, activation='softmax'))

# Compile model 
from tensorflow.keras.optimizers import Adam
model2.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model2.summary() 


# Step 4: Train the model and store the history

#from tensorflow.keras.callbacks import EarlyStopping

#ensures model does not train for unnecessary epochs and avoids overfitting 
#early_stopping = EarlyStopping(
   # monitor='val_loss',   # Monitor validation loss
   # patience=6,           
   # restore_best_weights=True  
#)


EPOCHS = 30

# Train the first model
history1 = model2.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Save the trained model
model2.save("model2_trained.keras")



# Plot accuracy and loss for Model2
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Train Accuracy (Model 2)')
plt.plot(history1.history['val_accuracy'], label='Val Accuracy (Model 2)')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Model 2 Accuracy")
plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'], label='Train Loss (Model 2)')
plt.plot(history1.history['val_loss'], label='Val Loss (Model 2)')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title("Model 2 Loss")

plt.show()







