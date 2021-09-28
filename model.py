# Import modules and methods
from rruff import download_all_rruff, preprocess_dataset, split_dataset, modified_LeNet

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.data import Dataset, AUTOTUNE

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from glob import glob
import warnings
import os

# Download RRUFF dataset
if not os.path.isdir('rruff_data'):
    download_all_ruff()

# Get all sample paths
raw_files = glob('rruff_data/*/*.txt')
# Get all RAW sample paths (no baseline corrected)
raw_raw_files = [file for file in raw_files if 'RAW' in file]
# Get all processed sample paths (baseline corrected)
raw_processed_files = [file for file in raw_files if 'Processed' in file]

# Preprocessing
# 0. Increase dataset size by augmenting samples with less than n measured spectra
# 1. Resample spectra to standardize array shapes
# 2. Reshape spectra into square 32x32x1 tensors
# 3. Encode categorical labels as unique integers
# 4. Split dataset into training, validation, and test sets
# 5. Build tensorflow datasets and define batch size, shuffle, and performance tuning
spectra, labels = preprocess_dataset(raw_processed_files)

# Print unique mineral spectra counts
minerals, counts = np.unique(labels, return_counts=True)
print('Augmented dataset has {} unique minerals with at least {} augmented spectra each'
      .format(len(counts), np.min(counts)))

# Encode categorical labels
label_encoder = LabelEncoder().fit(labels)
labels_encoded = label_encoder.transform(labels)

# Split dataset into training, validation, and test sets
train_spectra, train_labels, val_spectra, val_labels, test_spectra, test_labels \
  = split_dataset(spectra, labels_encoded, val_ratio=0.1, test_ratio=0.1)

# Build tensorflow datasets
train_ds = Dataset.from_tensor_slices((train_spectra, train_labels))
val_ds = Dataset.from_tensor_slices((val_spectra, val_labels))
test_ds = Dataset.from_tensor_slices((test_spectra, test_labels))

# Define batch and shuffle sizes
batch_size = 32
shuffle_size = 1000
train_ds = train_ds.shuffle(shuffle_size).batch(batch_size)
val_ds = val_ds.shuffle(shuffle_size).batch(batch_size)
test_ds = test_ds.shuffle(shuffle_size).batch(batch_size)

# Visualize first four training spectra
plt.figure(figsize=(9, 9))
for spectrum, label in train_ds.take(1):
    title = label_encoder.inverse_transform(label)
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(spectrum[i])
        plt.title(title[i])
        plt.tight_layout()
        plt.axis('off')

plt.show(block=False)
plt.savefig('spectra.png')

# Configuring the dataset for performance
autotune = AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=autotune)
val_ds = val_ds.cache().prefetch(buffer_size=autotune)

# Building sequential models
num_classes = len(label_encoder.classes_)
# Input shape is (batch size, rows/height, columns/width, channels)
input_shape = (batch_size, train_spectra.shape[1], train_spectra.shape[2], 1)

# Model 1: modified LeNet5
# https://arxiv.org/pdf/1708.09022.pdf
# Notes:
# - Uses ReLU
# - Uses dropout
# - Uses MaxPooling
# - Uses BatchNormalization
model = modified_LeNet(input_shape, num_classes)

# Compile model
model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Model summary
model.summary()

# Log training history to csv file
csv_logger = CSVLogger('modified_LeNet_history.csv', append=True)

# Stop model if validation accuracy doesn't improve after 5 epochs
early_stop = EarlyStopping(patience=15, restore_best_weights=True)

# Train model
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[csv_logger, early_stop]
)

# Visualize results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(11, 5.5))
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

plt.savefig('validation.png')
plt.show(block=False)

# Predict test set
results = model.evaluate(test_ds)
print('Model predicts test set with {:2.1f}% accuracy'.format(results[1]*100))
