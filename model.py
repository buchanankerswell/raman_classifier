# Import
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, LeakyReLU, Dropout, GlobalAveragePooling2D, MaxPooling1D, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Spectra preprocessing using keras API
batch_size = 16
data = tf.keras.preprocessing.image_dataset_from_directory(
    'spectra_images/processed/',
    color_mode='grayscale',
    image_size=(150,150),
    batch_size=batch_size,
    seed=42
)

# Model
num_labels = len(data.class_names)
model = Sequential([
    Conv2D(
        input_shape=(150,150,1),
        filters=128,
        kernel_size=20,
        padding='same',
        activation='relu',
    ),
    MaxPooling1D(pool_size=2, padding='valid'),
    Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'),
    MaxPooling1D(pool_size=2, padding='valid'),
    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(num_labels, activation='softmax')
])