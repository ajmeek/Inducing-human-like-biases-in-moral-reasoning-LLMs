#!/usr/bin/env python3

'''
fMRI_utils.py

Generates dataset for supervised learning with input text and related fMRI data.
'''

import bids
from pathlib import Path
from nilearn.masking import compute_epi_mask, apply_mask
import nibabel as nib
import numpy as np

datapath = Path('./data')

#loop to turn all functional mri data by subject into 2d arrays. This can then be flattened for use in BERT linear layer fine tuning
def generate_flattened_data():
    #download the dataset into the data folder, but don't add it to git
    fmri_path = datapath / "ds000212"

    #this data structure will allow us to capture anatomical or functional data dependent on run and subject
    layout = bids.BIDSLayout(fmri_path, config=['bids', 'derivatives'])
    for i in list(layout.get_subjects()):
        #print(layout.get(subject=i,extension='nii.gz'))
        subject_path = datapath / f"ds000212/sub-{i}/func/"
        target_path = datapath / f"functional_flattened/sub-{i}"
        target_path.mkdir(parents=True)
        k = 0
        for globf in subject_path.glob("*.nii.gz"):
            mask_img = compute_epi_mask(globf)
            masked_data = apply_mask(globf, mask_img)
            k = k+1
            filename = datapath / f"functional_flattened/sub-{i}/{k}.npy"
            #create file
            with open(filename, "w") as f:
                np.save(filename, masked_data)

def main():
    generate_flattened_data()

    #load out of the numpy files as so
    test = np.load(datapath / 'functional_flattened/sub-03/1.npy')
    print(test)

if __name__ == '__main__':
    assert datapath.exists()
    main()
