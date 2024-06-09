import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

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
    tf.keras.layers.Dense(len('categories'), activation='softmax')  # Adjust based on the number of your categories
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
    'image_directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    save_to_dir='output_directory',  # Save augmented images for inspection
    save_prefix='aug_'
)

# Fine-tune the model
history = model.fit(train_generator, epochs=10, validation_split=0.2)

# Optionally, save the fine-tuned model
model.save('fine_tuned_model.h5')
