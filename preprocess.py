# Imports
from sklearn.preprocessing import MinMaxScaler
from rruff import get_sample_from_file
from rruff import download_all_rruff
from rruff import RamanSample
from PIL import Image
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import contextlib
import pickle
import shutil
import time
import csv
import io
import os

# Download all RRUFF spectra from:
# https://rruff.info/zipped_data_files/raman/
print('Downloading RRUFF database')
download_all_rruff()
# Raw files list
raw_files = glob('data/*/*.txt')
print('Number of raw spectra:', len(raw_files))
# Parse filename and move spectra into new directory structure
# Note: this step is necessary to run keras ImageDataGenerator flow_from_directory method
# https://keras.io/api/preprocessing/image/
# Directory structure:
#  Processed spectra
#  |-- Mineral 1
#     |-- spectra
#     |-- spectra
#     |-- spectra
#  |-- Mineral 2
#     |-- spectra
#     |-- spectra
#     |-- spectra
#  Raw spectra
#  |-- Mineral 1
#     |-- spectra
#     |-- spectra
#     |-- spectra
#  |-- Mineral 2
#     |-- spectra
#     |-- spectra
#     |-- spectra
for fname in tqdm(raw_files, desc = 'Processing spectra '):
    # Split basename
    basename, ext = os.path.splitext(os.path.basename(fname))
    # Parse basename
    mineral, rruff_id, spectra_type, wavelength, rotation, orientation, data_status, unique_id = basename.split('__')
    # Path for new directory
    path = os.path.join('spectra_images', '{}/{}'.format(data_status.split('_')[-1].lower(), mineral.lower()))
    # Make directory
    os.makedirs(path, exist_ok=True)
    # Get sample
    spc = get_sample_from_file(fname).spectrum
    try:
        # Scale spectrum to range [0, 1] (normalizing)
        spc_scaled = MinMaxScaler().fit_transform(spc)
        # Save array as image for importing into keras ImageDataGenerator
        img = Image.fromarray(spc_scaled, 'L')
        img.save(path + '/' + basename + '.png')
    except ValueError:
        print('Something went wrong with sample ', basename)
        pass
