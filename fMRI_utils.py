import bids
from pathlib import Path
from nilearn.masking import compute_epi_mask, apply_mask
import nibabel as nib
import glob
import numpy as np
import os

#download the dataset into the data folder, but don't add it to git
fmri_path = Path("data/ds000212")

#this data structure will allow us to capture anatomical or functional data dependent on run and subject
layout = bids.BIDSLayout(fmri_path, config=['bids', 'derivatives'])


if not os.path.exists("data/functional_flattened"):
    os.makedirs("data/functional_flattened")

#loop to turn all functional mri data by subject into 2d arrays. This can then be flattened for use in BERT linear layer fine tuning
def generate_flattened_data():
    for i in list(layout.get_subjects()):
        #print(layout.get(subject=i,extension='nii.gz'))
        subject_path = "data/ds000212/sub-"+str(i)+"/func/"
        if not os.path.exists("data/functional_flattened/sub-"+str(i)):
            os.makedirs("data/functional_flattened/sub-"+str(i))
        k = 0
        for j in glob.iglob(subject_path+"*.nii.gz"):
            mask_img = compute_epi_mask(j)
            masked_data = apply_mask(j, mask_img)

            k = k+1
            filename = "data/functional_flattened/sub-"+str(i)+"/"+str(k)+".npy"

            #create file
            f = open(filename, "w")
            np.save(filename, masked_data)


generate_flattened_data()

#load out of the numpy files as so
test = np.load('data/functional_flattened/sub-03/1.npy')
print(test)