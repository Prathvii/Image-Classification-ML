import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt



# Define directories
train_dir = r'C:\Users\prath\zxc\tiny-imagenet-200\train'
test_dir = r'C:\Users\prath\zxc\tiny-imagenet-200\test'

# Create ImageDataGenerators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(64, 64),batch_size=32,class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,target_size=(64, 64),batch_size=32, class_mode='categorical')

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(200, activation='softmax')  # Change this to the number of classes
])
model.add(tf.keras.layers.Dropout(0.5))


# Compile the model
model.compile(optimizer='RMSprop', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

# Create a LearningRateScheduler callback
callback = LearningRateScheduler(scheduler)

# Compile the model
model.compile(optimizer='RMSprop', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[callback])
# Save the model
model.save(r'C:\Users\prath\zxc\notdone.h5')
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\nTest accuracy:', test_acc)



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
