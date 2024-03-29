DESCRIPTION = '''
This script converts .nii.gz into .npy files
'''

from datetime import datetime
from sys import argv
from pathlib import Path
import numpy as np
from csv import DictReader
from nilearn.masking import compute_epi_mask, apply_mask
import nilearn
import nilearn.maskers
import nibabel as nib
import argparse
from re import search
import torch as t


def main():
    a = get_args().parse_args()
    masked_data = apply_mask(a.brain_niigz, a.mask_niigz)
    # Reduce the size in 10 times:
    masked_data = t.nn.Conv1d(1, 1, kernel_size=10,stride=10)(masked_data.view(masked_data.shape[0], 1, -1))
    np.savez(a.to_npz, masked_data.view(masked_data.shape[0],-1))


def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('brain_niigz', type=Path)
    parser.add_argument('mask_niigz', type=Path)
    parser.add_argument('to_npz', type=Path,)
    return parser


if __name__ == '__main__':
    main()
