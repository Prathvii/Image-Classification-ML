import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to classify an image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions  # Get the top 3 class labels and their confidences

# Function to separate images into different categories and organize them into folders
def organize_images(image_paths, output_directory):
    categories = {
        "person": "Human",
        "car": "Car",
        "animal": "Animal",
        "food": "Food",
        "building": "Building",
        "flower": "Flower",
        "electronic": "Electronic",
        "sports": "Sports",
        "nature": "Nature",
        "music": "Music",
        "art": "Art",
        "book": "Book",
        "fashion": "Fashion",
        "vehicle": "Vehicle",
        "fruit": "Fruit",
        "insect": "Insect",
        "technology": "Technology",
        "landscape": "Landscape",
        "tool": "Tool",
        "furniture": "Furniture",
        "toy": "Toy",
        "sky": "Sky",
        "water": "Water",
        "portrait": "Portrait",
        "mountain": "Mountain",
        "beach": "Beach",
        "cityscape": "Cityscape",
        "dessert": "Dessert",
        "architecture": "Architecture",
        "cuisine": "Cuisine",
        "gadget": "Gadget",
        "wildlife": "Wildlife",
        "jewelry": "Jewelry",
        "entertainment": "Entertainment",
        "garden": "Garden",
        "holiday": "Holiday",
        "medical": "Medical",
        "astronomy": "Astronomy",
        "event": "Event",
        "exercise": "Exercise",
        "industrial": "Industrial",
        "fantasy": "Fantasy",
        "family": "Family",
        "party": "Party",
        "plant": "Plant",
        "science": "Science",
        "season": "Season",
        "texture": "Texture"
        # Add more classes as needed
    }

    # Create folders for each category
    for category_folder in categories.values():
        os.makedirs(os.path.join(output_directory, category_folder), exist_ok=True)

    # Organize images into folders
    for img_path in image_paths:
        predicted_labels = classify_image(img_path)

        for _, label, _ in predicted_labels:
            for category_label, category_folder in categories.items():
                if category_label in label.lower():
                    destination_folder = os.path.join(output_directory, category_folder)
                    shutil.copy(img_path, destination_folder)
                    break

# Directory containing images
image_directory = '/path/to/your/images'

# Output directory where the organized folders will be created
output_directory = '/path/to/your/output'

# Get a list of image file paths in the directory
image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Organize images into folders
organize_images(image_paths, output_directory)
