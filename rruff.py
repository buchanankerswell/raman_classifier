# Methods for downloading and preprocessing
# the RRUFF database for tensorflow/keras workflow
# Note: inspired by Derek Kanes
# https://github.com/DerekKaknes/raman/
# after Liu et al. (2017)
# https://arxiv.org/abs/1708.09022

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LeakyReLU, Activation, Flatten

from sklearn.model_selection import train_test_split
from urllib.request import Request, urlopen
import scipy.ndimage as sp_filter
from collections import Counter
from bs4 import BeautifulSoup
from math import ceil, floor
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
import numpy as np
import contextlib
import warnings
import random
import re
import os

# Method for downloading and extracting data directly from RRUFF
# https://rruff.info/zipped_data_files/raman/
def _download_rruff_from_url(url):
    # Create directory if it doesn't exist
    if not os.path.exists('rruff_data'):
        print('Creating directory ./ruff_data')
        os.makedirs('rruff_data')
    # Extract RRUFF spectrum type
    dir_name = 'rruff_data/' + re.split('/', url)[-1].split('.')[0]
    # Open, read, and extract zip into directory
    with urlopen(url) as response:
        print('Reading', url)
        with ZipFile(BytesIO(response.read())) as zpfile:
            # Print file size
            size = sum([zinfo.file_size for zinfo in zpfile.filelist])
            print('Files:', len(zpfile.infolist()))
            print('Total size: ', round(size / 1e6, 2), 'Mb', sep='')
            for member in tqdm(zpfile.infolist(), desc='Extracting'):
                try:
                    zpfile.extract(member, dir_name)
                except ValueError:
                    warnings.warn('Extraction failed for: {}'.format(member), Warning)

# Download all RRUFF data from urls
def download_all_rruff():
    # Create directory if it doesn't exist
    if not os.path.exists('rruff_data'):
        print('Creating directory ./rruff_data')
        os.makedirs('rruff_data')
    # RRUFF url
    url = 'https://rruff.info/zipped_data_files/raman/'
    # Request page
    req = Request(url)
    # Open page
    html_page = urlopen(req)
    # Parse links with BS object
    soup = BeautifulSoup(html_page, 'html.parser')
    # Get links
    links = []
    for link in soup.findAll('a'):
        links.append(link.get('href'))
    zip_idx = [
        i for i, word in enumerate(links) \
        if word.endswith('zip') and not word.startswith('LR')
    ]
    zip_links = [links[i] for i in zip_idx]
    # Construct urls
    zip_urls = [url + lnk for lnk in zip_links]
    # Print links
    print('Found urls:', *zip_urls, sep='\n')
    # Download, and extract
    for url in zip_urls:
        _download_rruff_from_url(url)

# Method for parsing lines from RRUFF .txt files
def _parse_line(l):
    if l.strip() == '':
        return None, None
    elif l.startswith('##'):
        k_raw, v = l.split('=')[:2]
        k = k_raw[2:].lower().replace(' ', '_')
        if k == 'names':
            k = 'mineral'
        if k == 'pin_id':
            k = 'pin'
        if k == 'end':
            return None, None
    else:
        k = 'spectrum'
        try:
            x_val, y_val = [float(x.strip()) for x in l.split(', ')]
        except ValueError:
            v = []
        else:
            v = [x_val, y_val]
    return k, v

# Method for parsing whole RRUFF .txt files
def _parse_raw_file(file_path):
    attrs = {'spectrum' : []}
    with open(file_path) as f:
        c = 0
        for line in f:
            c += 1
            k, v = _parse_line(line.strip())
            if k:
                if k == 'spectrum':
                    attrs[k].append(v)
                else:
                    attrs[k] = v
    if None in attrs['spectrum']:
        attrs['spectrum'] = None
    else:
        attrs['spectrum'] = np.array(attrs['spectrum'])
    return attrs

# Method for reading  RRUFF .txt file and
# saving as numpy ndarray with label
def get_spectrum_from_file(file_path):
    attrs = _parse_raw_file(file_path)
    spectrum = attrs.get('spectrum')
    try:
        if spectrum.size == 0:
            raise ValueError('No spectrum exists')
    except ValueError:
        raise
    else:
        label = attrs.get('mineral')
        return (spectrum, label)

# Method for resampling spectrum at equally spaced intervals
# Note: returns 1D array
def resample_spectrum(spectrum, resample_num=512, wavenumbers=False):
    wavenumber = spectrum[:,0]
    intensity = spectrum[:,1]
    wavenumber_begin = ceil(min(wavenumber))
    wavenumber_end = floor(max(wavenumber))
    wavenumber_range = np.linspace(wavenumber_begin, wavenumber_end, num=resample_num)
    intensity_resample = np.interp(wavenumber_range, wavenumber, intensity)
    if wavenumbers:
        return(np.vstack((wavenumber_range, intensity_resample)).T)
    else:
        return(intensity_resample)

# Method for augmenting spectra
# Note: randomly shifts wavenumbers and
# randomly scales intensities
def augment_spectrum(spectrum, shift=3, scale=0.2, kernel_width=10, wavenumbers=False):
    wavenumber = spectrum[:,0]
    intensity = spectrum[:,1]
    # Shift wavenumbers randomly
    wavenumber_shifted = wavenumber + random.uniform(-shift, shift)
    # Scale intensity randomly
    conv_weights = [random.uniform(-scale*0.1, scale) for _ in range(kernel_width)]
    intensity_scaled = sp_filter.convolve1d(intensity, weights=conv_weights)
    if wavenumbers:
        return(np.vstack((wavenumber_shifted, intensity_scaled)).T)
    else:
        return(intensity_scaled)

# Method for preprocessing dataset
# 0. Increase dataset size by augmenting samples with less than n measured spectra
# 1. Resample spectra to standardize array shapes
# 2. Reshape spectra into square 32x32x1 tensors
def preprocess_dataset(raw_files, shift=3, scale=0.2, kernel_width=10, wavenumbers=False, augment_count=50):
    # Get unique mineral counts
    mins_list = sorted([os.path.split(file)[1].split('__')[0] for file in raw_files])
    mins, counts = np.unique(mins_list, return_counts=True)
    # Initialize lists
    spc = []
    label = []
    # Get sample
    pbar = tqdm(
        mins,
        desc='Processing unique minerals',
        bar_format='{desc}:{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
    )
    for mineral in pbar:
        # Get all measured spectra paths for a specific mineral
        sample_paths = [file for file in raw_files if mineral in file]
        # Randomly sample (with replacement) k spectra and augment
        for file in random.choices(sample_paths, k=augment_count):
            try:
                smp = get_spectrum_from_file(file)
            except ValueError:
                # Skip sample if no spectrum exists
                warnings.warn('No spectrum exists for {}'.format(file), Warning)
            else:
                # Augment spectrum
                spc_augmented = augment_spectrum(smp[0], shift, scale, kernel_width, True)
                # Subsampling resolution is 512 for 2D array with wavenumbers
                if wavenumbers:
                    spc_resampled = resample_spectrum(spc_augmented, 512, True)
                # Otherwise use 1024 for 1D array with only intensity signal
                else:
                    spc_resampled = resample_spectrum(spc_augmented, 1024)
                # Scale spectrum
                spc_scaled = spc_resampled / spc_resampled.max(axis=0)
                # Add channel dimension
                spc_expanded = np.expand_dims(spc_scaled, axis=-1)
                # Append to list
                spc.append(spc_expanded)
                label.append(mineral)
    # Turn lists into numpy arrays
    spc = [np.reshape(spectrum, (32,32, 1), 'C') for spectrum in spc]
    spc = np.stack(spc)
    label = np.array(label)
    return(spc, label)

# Method for splitting training/test set
def split_dataset(spectra, labels, test_ratio=0.15, val_ratio=0.15):
    # Number of test spectra
    n_test = ceil(len(spectra)*test_ratio)
    # Split into training, validation, and test sets
    test_size = test_ratio
    val_size = val_ratio/(1-test_ratio)
    print('Splitting dataset into {:.0f}% train, {:.0f}% val, and {:.0f}% test sets'.format(
        (1-(test_ratio+val_ratio))*100,
        val_ratio*100,
        test_ratio*100
    ))
    train_spectra, test_spectra, train_labels, test_labels \
        = train_test_split(spectra, labels, test_size=test_size)
    train_spectra, val_spectra, train_labels, val_labels \
        = train_test_split(spectra, labels, test_size=val_size)
    print('Train: {} spectra with {} classes\nValidation: {} spectra with {} classes\nTest: {} spectra with {} classes'.format(
        len(train_spectra),
        len(np.unique(train_labels)),
        len(val_spectra),
        len(np.unique(val_labels)),
        len(test_spectra),
        len(np.unique(test_labels))
    ))
    return(train_spectra, train_labels, val_spectra, val_labels, test_spectra, test_labels)

# Model 1: modified LeNet5-2D
# https://arxiv.org/pdf/1708.09022.pdf
# Notes:
# - Uses ReLU
# - Uses dropout
# - Uses MaxPooling
# - Uses BatchNormalization
def modified_LeNet(input_shape, num_classes):
    return(
        Sequential([
            Conv2D(16, 5, input_shape=input_shape[1:], data_format='channels_last'),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling2D(2),
            Conv2D(32, 5),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling2D(2),
            Conv2D(64, 5),
            BatchNormalization(),
            LeakyReLU(),
            Flatten(),
            Dense(2048),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.5),
            Dense(num_classes),
            BatchNormalization(),
            Activation('softmax')
        ])
    )
