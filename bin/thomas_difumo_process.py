DESCRIPTION = '''
This script converts .pyd files into .npy files
suitable for fine tuning a language model.
'''
from re import search
import argparse
import nibabel as nib
import nilearn.maskers
import nilearn
from nilearn.masking import compute_epi_mask, apply_mask
from csv import DictReader
import numpy as np
from pathlib import Path
from sys import argv
from datetime import datetime
import tarfile
import os
from os.path import isfile, join


def uncompress_tar_files() -> None:
    """

    """

    try:
        os.mkdir('../data/thomas_ds000212_difumo/uncompressed')
        #create the uncompressed folder if not already there
    except:
        pass

    if os.listdir('../data/thomas_ds000212_difumo/uncompressed'):
        #directory not empty, already compressed
        return

    data_dir = Path(__file__).parent.parent
    data_dir = str(data_dir) + '/data/thomas_ds000212_difumo'
    data_dir = Path(data_dir)

    directory_contents = os.listdir(path=data_dir)
    only_files = [f for f in directory_contents if isfile(join(data_dir, f))]

    for f in only_files:
        run = tarfile.open('../data/thomas_ds000212_difumo/' + f)
        run.extractall('../data/thomas_ds000212_difumo/uncompressed')
        run.close()

uncompress_tar_files()