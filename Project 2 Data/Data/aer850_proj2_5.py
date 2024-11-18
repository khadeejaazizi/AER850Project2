# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:21:22 2024

@author: Khadeeja Azizi
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Load the saved model
model2 = load_model("model2_trained.keras")

# Define paths to test images
test_images = {
    "test_crack": "test/crack/test_crack.jpg",
    "test_missinghead": "test/missing-head/test_missinghead.jpg",
    "test_paintoff": "test/paint-off/test_paintoff.jpg"
}

# Class label mapping
class_labels = {0: 'crack', 1: 'missing-head', 2: 'paint-off'}  

# Function to display true label, predicted label, and probabilities
def display_prediction_with_probabilities(img_path, model, true_label):
  
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values

    # Predict the class
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    probabilities = predictions[0] * 100  # Convert to percentages

    # Plot the image with annotations
    plt.imshow(image.load_img(img_path, target_size=(500, 500)))
    plt.axis('off')

    # Display true and predicted labels above the image
    plt.title(f"True Classification Label: {true_label}\nPredicted Classification Label: {predicted_label}", fontsize=12, pad=20)

    # Overlay the probabilities on the image
    for i, (label, prob) in enumerate(class_labels.items()):
        plt.text(11, 470 -i *40, f"{prob}: {probabilities[i]:.1f}%", color="blue",fontsize=20)

    plt.show()

# Run predictions and display results with true labels
display_prediction_with_probabilities(test_images['test_paintoff'], model2, true_label="paint-off")

display_prediction_with_probabilities(test_images['test_crack'], model2, true_label="crack")

display_prediction_with_probabilities(test_images['test_missinghead'], model2, true_label="missing-head")
