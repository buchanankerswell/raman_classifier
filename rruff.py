__all__ = [
    'download_all_rruff',
    'download_rruff_from_url',
    'create_from_file',
    'get_spectrum_and_label_from_file'
]

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from zipfile import ZipFile
from zipfile import error
from io import BytesIO
from tqdm import tqdm

import numpy as np
import warnings
import glob
import sys
import re
import os

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
    with urlopen(url) as zp:
        print('Reading', url)
        with ZipFile(BytesIO(zp.read())) as zpfile:
            # Print file size
            size = sum([zinfo.file_size for zinfo in zpfile.filelist])
            print('Files:', len(zpfile.infolist()))
            print('Total size: ', round(size / 1e6, 2), 'Mb', sep='')
            for member in tqdm(zpfile.infolist(), desc='Extracting '):
                try:
                    zpfile.extract(member, dir_name)
                except error as e:
                    pass
    print('Done!')

# Download all RRUFF data from urls
def download_all_rruff():
    # Create directory if it doesn't exist
    if not os.path.exists('rruff_data'):
        print('Creating directory ./ruff_data')
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
    zip_idx = [i for i, word in enumerate(links) if word.endswith('zip') and not word.startswith('LR')]
    zip_links = [links[i] for i in zip_idx]
    # Construct urls
    zip_urls = [url + lnk for lnk in zip_links]
    # Print links
    print('Found files:', *zip_urls, sep='\n')
    # Download, and extract
    for url in zip_urls:
        download_rruff_from_url(url)

# Define RamanSample class
class RamanSample(object):
    def __init__(self, mineral, rruffid, spectrum, ideal_chemistry=None,
            locality=None, owner=None, source=None,
            description=None, status=None, url=None, measured_chemistry=None):
        self.mineral = mineral
        self.rruffid = rruffid
        self.spectrum = spectrum
        self.ideal_chemistry = ideal_chemistry
        self.locality = locality
        self.owner = owner
        self.source = source
        self.description = description
        self.status = status
        self.url = url
        self.measured_chemistry = measured_chemistry

# Method for parsing lines from RRUFF .txt files
def _parse_line(l):
    if l.strip() == "":
        return None, None
    elif l.startswith("##"):
        k_raw, v = l.split("=")[:2]
        k = k_raw[2:].lower().replace(" ", "_")
        if k == "names":
            k = "mineral"
        if k == "end":
            return None, None
    else:
        k = "spectrum"
        try:
            x_val, y_val = [float(x.strip()) for x in l.split(", ")]
        except ValueError:
            warnings.warn("Could not convert to string: {}".format(l))
            x_val, y_val = (float(l.split(", ")[0]), 0)
        v = np.array([x_val, y_val])

    return k, v

# Method for parsing whole RRUFF .txt files
def _parse_raw_file(file_path):
    attrs = {"spectrum" : []}
    with open(file_path) as f:
        c = 0
        for line in f:
            c +=1
            k, v = _parse_line(line.strip())
            if k:
                if k == "spectrum":
                    attrs[k].append(v)
                else:
                    attrs[k] = v
    attrs['spectrum'] = np.array(attrs['spectrum'])
    return attrs

# Method for reading  RRUFF .txt file and
# saving as numpy ndarray with label
def get_spectrum_and_label_from_file(file_path):
    attrs = _parse_raw_file(file_path)
    spectrum = attrs.get("spectrum")
    label = attrs.get("mineral")
    return (spectrum, label)

# Method for reading  RRUFF .txt file and
# saving as RamanSample class object
def create_from_file(file_path):
    attrs = _parse_raw_file(file_path)
    return RamanSample(**attrs)

if __name__ == "__main__":
    download_all_rruff()
