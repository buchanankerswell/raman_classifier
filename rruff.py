# Methods for downloading and preprocessing
# the RRUFF database for tensorflow/keras workflow
# Note: inspired by Derek Kanes
# https://github.com/DerekKaknes/raman/
# after Liu et al. (2017)
# https://arxiv.org/abs/1708.09022

from sklearn.model_selection import train_test_split
from urllib.request import Request, urlopen
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
def resample_spectrum(spectrum, resample_num=512):
    wavenumber = spectrum[:,0]
    intensity = spectrum[:,1]
    wavenumber_begin = ceil(min(wavenumber))
    wavenumber_end = floor(max(wavenumber))
    wavenumber_range = np.linspace(wavenumber_begin, wavenumber_end, num=resample_num)
    intensity_resample = np.interp(wavenumber_range, wavenumber, intensity)
    return(np.vstack((wavenumber_range, intensity_resample)).T)

# Method for augmenting spectra
# Note: randomly shifts wavenumbers and
# randomly scales intensities
def augment_spectrum(spectrum, shift=3, scale=0.2):
    wavenumber = spectrum[:,0]
    intensity = spectrum[:,1]
    # Shift wavenumbers randomly
    wavenumber_shifted = wavenumber + random.randint(-shift, shift)
    # Scale intensity randomly
    intensity_scaled = intensity + (intensity * random.uniform(-scale, scale))
    return(np.vstack((wavenumber_shifted, intensity_scaled)).T)

# Method for preprocessing dataset
def preprocess_dataset(raw_files, resample_num=512, shift=3, scale=0.2, augment_count=50):
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
        sample_paths = [file for file in raw_files if mineral in file]
        for file in random.choices(sample_paths, k=augment_count):
            try:
                smp = get_spectrum_from_file(file)
            except ValueError:
                # Skip sample if no spectrum exists
                warnings.warn('No spectrum exists for {}'.format(file), Warning)
            else:
                spc_augmented = augment_spectrum(spectrum=smp[0], shift=shift, scale=scale)
                spc_resampled = resample_spectrum(spc_augmented, resample_num=resample_num)
                spc_scaled = spc_resampled / spc_resampled.max(axis=0)
                spc.append(spc_scaled)
                label.append(mineral)
    # Turn lists into numpy arrays
    spc = [np.expand_dims(np.reshape(spc[i], (32,32), 'C'), axis=-1) for i in range(len(spc))]
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
