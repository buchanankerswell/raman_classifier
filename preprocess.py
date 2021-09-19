# Methods for downloading and preprocessing
# the RRUFF database for tensorflow/keras workflow
# Note: much of this code was developed from Derek Kanes
# https://github.com/DerekKaknes/raman/
# after Liu et al. (2017)
# https://arxiv.org/abs/1708.09022

__all__ = [
    'download_all_rruff',
    'download_rruff_from_url',
    'create_from_file',
    'get_spectrum_and_label_from_file',
    'convert_spectrum_to_image'
]

from sklearn.preprocessing import MinMaxScaler
from urllib.request import Request, urlopen
from skimage.transform import resize
from bs4 import BeautifulSoup
from zipfile import ZipFile
from zipfile import error
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from glob import glob
from math import ceil
import numpy as np
import contextlib
import warnings
import shutil
import random
import sys
import png
import io
import re
import os

# Define RamanSample class
class RamanSample(object):
    def __init__(self, mineral, rruffid, spectrum, ideal_chemistry=None,
            locality=None, owner=None, source=None, orientation=None,
            description=None, status=None, url=None, pin=None, measured_chemistry=None):
        self.mineral = mineral
        self.rruffid = rruffid
        self.spectrum = spectrum
        self.ideal_chemistry = ideal_chemistry
        self.locality = locality
        self.owner = owner
        self.source = source
        self.orientation = orientation
        self.description = description
        self.status = status
        self.url = url
        self.pin = pin
        self.measured_chemistry = measured_chemistry

# Method for downloading and extracting data directly from RRUFF
# https://rruff.info/zipped_data_files/raman/
def download_rruff_from_url(url):
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
        download_rruff_from_url(url)

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
            warnings.warn('Could not parse line: {}'.format(l), Warning)
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
        warnings.warn('No spectrum exists', Warning)
        attrs['spectrum'] = None
    else:
        attrs['spectrum'] = np.array(attrs['spectrum'])
    return attrs

# Method for reading  RRUFF .txt file and
# saving as numpy ndarray with label
def get_spectrum_and_label_from_file(file_path):
    attrs = _parse_raw_file(file_path)
    spectrum = attrs.get('spectrum')
    label = attrs.get('mineral')
    if spectrum.size == 0:
        warnings.warn('No spectrum exists', Warning)
    return (spectrum, label)

# Method for reading  RRUFF .txt file and
# saving as RamanSample class object
def get_sample_from_file(file_path):
    attrs = _parse_raw_file(file_path)
    if attrs['spectrum'].size == 0:
        warnings.warn('No spectrum exists', Warning)
    return RamanSample(**attrs)

# Method for converting spectra into 16-bit png images
# Note: this step is necessary to run keras ImageDataGenerator flow_from_directory method
# https://keras.io/api/preprocessing/image/
def convert_spectrum_to_image(file_path):
    # Get spectrum
    spc = get_sample_from_file(file_path).spectrum
    # If no spectrum exists skip processing
    if spc.size == 0:
        warnings.warn('No spectrum exists for file: {}'.format(file_path), Warning)
    else:
        # Split file path
        basename, ext = os.path.splitext(os.path.basename(file_path))
        # Parse basename of file path
        mineral, \
        rruff_id, \
        spectra_type, \
        wavelength, \
        rotation, \
        orientation, \
        data_status, \
        unique_id = basename.split('__')
        # Create path to new directory
        new_dir = 'training_images/{}'.format(mineral).lower()
        # Make new directory if it doesn't exist
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir, exist_ok=True)
        # Normalizing spectrum
        # Scale to range [0,1]
        spc_scaled = MinMaxScaler(feature_range=(0,1)).fit_transform(spc)
        # Resize array
        spc_resize = resize(spc_scaled, output_shape=(150,150), anti_aliasing='True')
        # Convert array from float64 to 16 bit unsigned integers for better png image
        spc_int = (65535*((spc_resize-spc_resize.min())/spc_resize.ptp())).astype(np.uint16)
        # Save as png image
        img_path = os.path.join(new_dir,basename.lower()) + '.png'
        with open(img_path, 'wb') as f:
            writer = png.Writer(
                width=spc_int.shape[1],
                height=spc_int.shape[0],
                bitdepth=16,
                greyscale=True
            )
            img = spc_int.tolist()
            writer.write(f, img)

# Method for splitting training/test set
def train_test_split(test_ratio):
    # Make new directory for test set if it doesn't exist
    if not os.path.isdir('test_images'):
        os.makedirs('test_images', exist_ok=True)
    # Get training image paths
    train_images = glob('training_images/*/*.png')
    # Number of test images
    n = ceil(len(train_images)*test_ratio)
    # Move files
    for file in tqdm(random.sample(train_images, n), desc='Splitting training/test data'):
        # Split file path
        basename, ext = os.path.splitext(os.path.basename(file))
        # Parse basename of file path
        mineral, \
        rruff_id, \
        spectra_type, \
        wavelength, \
        rotation, \
        orientation, \
        data_status, \
        unique_id = basename.split('__')
        # Create path to new directory
        new_dir = 'test_images/{}'.format(mineral).lower()
        # Make new directory if it doesn't exist
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir, exist_ok=True)
        img_path = os.path.join(new_dir, basename) + '.png'
        shutil.move(file, img_path)

# Running as script
if __name__ == '__main__':
    # Download all RRUFF spectra from:
    # https://rruff.info/zipped_data_files/raman/
    print('Downloading RRUFF database ...')
    download_all_rruff()
    # Raw files list
    raw_files = glob('rruff_data/*/*.txt')
    print('Number of raw spectra:', len(raw_files))
    # Convert all spectra to b&w images and
    # move into new data structure
    for file_path in tqdm(raw_files, desc = 'Processing spectra'):
        convert_spectrum_to_image(file_path)
    # Split training/test sets 80/20
    train_test_split(0.2)
    print('Done!')

