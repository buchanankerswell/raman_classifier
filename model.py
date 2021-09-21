# Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# Load training, validation, and test sets
batch_size = 32
img_height = 150
img_width = 150
train_ds = image_dataset_from_directory(
    'training_images',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)
val_ds = image_dataset_from_directory(
    'training_images',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)
test_ds = image_dataset_from_directory(
    'test_images',
    seed=42,
    image_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)

# Visualize dataset
class_names = train_ds.class_names
plt.figure(figsize=(7, 7))
# Plot first 9 images
for image, label in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[i].numpy().astype("uint8"))
    plt.title(class_names[label[i]])
    plt.axis("off")

plt.show()

# Configuring the dataset for performance
autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=autotune)
val_ds = val_ds.cache().prefetch(buffer_size=autotune)

# Build sequential model
num_classes = len(class_names)
model = Sequential([
    Conv2D(16, 3, input_shape=(img_height,img_width,1), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)
])

# Compile model
model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(7, 3.5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper left')
plt.title('Training and Validation Loss')
plt.show()

# Improving training by augmenting spectra and by
# random image mutations
data_augmentation = Sequential([
    RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )
])

# Visualize augmented spectrum
plt.figure(figsize=(7, 7))
for image, _ in train_ds.take(1):
  for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0].numpy().astype("uint8"))
    plt.axis("off")

plt.show()

# Build sequential model with augmentation and dropout
# to improve model accuracy
num_classes = len(class_names)
model = Sequential([
    RandomTranslation(height_factor = 0.01, width_factor = 0.01),
    Conv2D(16, 3, input_shape=(img_height,img_width,1), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)
])

# Compile model
model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(7, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predict first 9 test images
predictions = model.predict(test_ds.take(1))
scores = tf.nn.softmax(predictions)

plt.figure(figsize=(7, 7))
# Plot first 9 images
for image, label in test_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[i].numpy().astype("uint8"))
    plt.title(class_names[np.argmax(score[i])])
    plt.axis("off")

plt.show()
