DESCRIPTION = '''
This script converts .nii.gz into .npy files
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


def main():
    a = get_args().parse_args()
	masked_data = apply_mask(a.brain_niigz, a.mask_niigz)
	np.savez(a.to_npz, masked_data)


def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('brain_niigz', type=Path)
    parser.add_argument('mask_niigz', type=Path)
    parser.add_argument('to_npz', type=Path,)
    return parser


if __name__ == '__main__':
    main()
