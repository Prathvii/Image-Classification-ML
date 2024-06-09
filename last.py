import os
import shutil
import numpy as np
import tensorflow as tf
from sympy import categories
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Load MobileNetV2 model without top layers (include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for classification
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(categories), activation='softmax')  # Adjust based on the number of your categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Organize images into folders
train_generator = datagen.flow_from_directory(
    output_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    save_to_dir=output_directory,  # Save augmented images for inspection
    save_prefix='aug_'
)

# Fine-tune the model
history = model.fit(train_generator, epochs=10, validation_split=0.2)

# Optionally, save the fine-tuned model
model.save('fine_tuned_model.h5')
