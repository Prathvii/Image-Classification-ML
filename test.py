from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model(r'C:\Users\prath\zxc\notdone.h5')

# Load an image file to test, resizing it to 64x64 pixels (as required by this model)
img = image.load_img(r"C:\Users\prath\OneDrive\Pictures\A51.jpg", target_size=(64, 64))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
img_array = np.expand_dims(img_array, axis=0)

# Normalize the image
img_array /= 255.

# Use the model to make a prediction
prediction = model.predict(img_array)

# Get the highest scoring class
prediction = np.argmax(prediction)

print("Predicted class:", prediction)



