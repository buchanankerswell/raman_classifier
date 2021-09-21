# Methods for downloading and preprocessing
# the RRUFF database for tensorflow/keras workflow
# Note: much of this code was developed from Derek Kanes
# https://github.com/DerekKaknes/raman/
# after Liu et al. (2017)
# https://arxiv.org/abs/1708.09022

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from glob import glob
from math import ceil
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

# Method for padding 2D array to specific shape
def _pad_array_to_specific_shape(array, new_height, new_width):
    h = array.shape[0]
    w = array.shape[1]
    top = (new_height - h) // 2
    bottom = new_height - top - h
    left = (new_width - w) // 2
    right = new_width - left - w
    return np.pad(
        array,
        pad_width=((top, bottom), (left, right)),
        mode='constant',
        constant_values=np.nan
    )

# Method for reading  RRUFF .txt file and
# saving as numpy ndarray with label
def _get_spectrum_from_file(file_path):
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

# Method for normalizing spectra by padding to specified length
# and scaling to a specified range
def _normalize_spectrum(spectrum, padded_length=8000, scaled_range=(0,1)):
    try:
        if spectrum.size == 0:
            raise ValueError('No spectrum exists')
    except ValueError:
        raise
    else:
        # Normalize spectrum
        spc_length = spectrum.shape[0]
        spc_padded = _pad_array_to_specific_shape(spectrum, padded_length, 2)
        # Scale to range [0,1]
        spc_scaled = MinMaxScaler(feature_range=scaled_range).fit_transform(spc_padded)
        return(spc_scaled)

# Method for splitting training/test set
def normalize_and_split_dataset(data_dir, test_ratio=0.15, val_ratio=0.15):
    # All samples
    raw_files = glob('{}/*/*.txt'.format(data_dir))
    # List and indexes
    norm_spc = []
    labels = []
    # Read and normalize each sample
    for file in tqdm(raw_files, desc='Processing spectra'):
        try:
            smp = _get_spectrum_from_file(file)
        except ValueError:
            warnings.warn('No spectrum exists ... skipping sample: {}'.format(file), Warning)
        else:
            norm_spc.append(_normalize_spectrum(smp[0]))
            labels.append(smp[1])
    # Turn lists into numpy arrays
    norm_spc = np.stack(norm_spc)
    labels = np.array(labels)
    # Number of test images
    n_test = ceil(len(norm_spc)*test_ratio)
    # Split into training, validation, and test sets
    test_size = test_ratio
    val_size = val_ratio/(1-test_ratio)
    print('Splitting dataset into {:.0f}% train, {:.0f}% val, and {:.0f}% test sets'.format(
        (1-(test_ratio+val_ratio))*100,
        val_ratio*100,
        test_ratio*100
    ))
    train_spectra, test_spectra, train_labels, test_labels \
        = train_test_split(
            norm_spc,
            labels,
            test_size=test_size,
            random_state=19
        )
    train_spectra, val_spectra, train_labels, val_labels \
        = train_test_split(
            norm_spc,
            labels,
            test_size=val_size,
            random_state=19
        )
    print('Train: {} spectra with {} classes\nValidation: {} spectra with {} classes\nTest: {} spectra with {} classes'.format(
        len(train_spectra),
        len(np.unique(train_labels)),
        len(val_spectra),
        len(np.unique(val_labels)),
        len(test_spectra),
        len(np.unique(test_labels))
    ))
    # Saving as compressed numpyz file (.npz)
    np.savez_compressed(
        'processed_spectra',
        train_spectra=train_spectra,
        train_labels=train_labels,
        val_spectra=val_spectra,
        val_labels=val_labels,
        test_spectra=test_spectra,
        test_labels=test_labels
    )

# Running as script
if __name__ == '__main__':
    # Download all RRUFF spectra from:
    # https://rruff.info/zipped_data_files/raman/
    print('Downloading RRUFF database ...')
    download_all_rruff()
    normalize_and_split_dataset('rruff_data')
    print('Done!')

