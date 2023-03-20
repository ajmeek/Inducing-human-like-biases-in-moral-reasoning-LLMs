#!/usr/bin/env python3

'''
fMRI_utils.py

Generates dataset for supervised learning with input text and related fMRI data.
'''

import bids
from pathlib import Path
import nilearn
import nibabel as nib
import numpy as np
from subprocess import run

datapath = Path('./data')

'''
This will take raw fMRI data and then apply the difumo mask, saving to the ds000212_difumo directory. 
It will save to numpy arrays of shape samples x dimensions, where dimensions is the number of ROI that the difumo mask has.
Currently this has 64 ROI (regions of interest). DiFuMo can go up to 1024, but applying that mask uses more RAM
than even google colab has, which is an issue.
'''
def apply_atlas():

    difumo_atlas = nilearn.datasets.fetch_atlas_difumo(dimension=64)
    print('DiFuMo has 64 ROIs')

    difumo_masker = nilearn.maskers.NiftiMapsMasker(difumo_atlas['maps'], resampling_target='data', detrend=True).fit() #actually it kills it - not enough memory

    fmri_path = datapath / "ds000212"
    layout = bids.BIDSLayout(fmri_path, config=['bids', 'derivatives'])
    for i in list(layout.get_subjects()):
        subject_path = datapath / f"ds000212/sub-{i}/func/"
        target_path = datapath / f"ds000212_difumo/sub-{i}"
        target_path.mkdir(parents=True, exist_ok=True)
        k = 0

        for bold_f in subject_path.glob("*.nii.gz"):
            k = k+1
            info(f'Processing {bold_f.name}')

            #print(list(subject_path.glob("*.nii.gz"))) #list of posix paths for all .nii.gz files in the subject's directory
            data = nib.load(bold_f) #no errors from this !
            roi_time_series = difumo_masker.transform(data)

            filename = target_path / f'{k}.npy'
            f = open(filename, "w")
            np.save(filename, roi_time_series)


#loop to turn all functional mri data by subject into 2d arrays. This can then be flattened for use in BERT linear layer fine tuning
def generate_flattened_data():
    #download the dataset into the data folder, but don't add it to git
    fmri_path = datapath / "ds000212"
    scenarios = np.loadtxt(
        datapath / "scenarios.csv", 
        delimiter=',',
        quotechar='"', 
        dtype= [
            ('item', 'S20'),
            ('type','U9999'),
            ('background', 'U9999'),
            ('action','U9999'),
            ('outcome','U9999'),
            ('accidental','U9999'),
            ('intentional','U9999'),
        ]
    ) 

    #this data structure will allow us to capture anatomical or functional data dependent on run and subject
    layout = bids.BIDSLayout(fmri_path, config=['bids', 'derivatives'])
    for i in list(layout.get_subjects()):
        #print(layout.get(subject=i,extension='nii.gz'))
        subject_path = datapath / f"ds000212/sub-{i}/func/"
        target_path = datapath / f"functional_flattened/sub-{i}"
        target_path.mkdir(parents=True, exist_ok=True)
        k = 0
        for bold_f in subject_path.glob("*.nii.gz"):
            info(f'Processing {bold_f.name}')
            #   bold_symlink_f = str(bold_f.resolve())
            #   mask_img = compute_epi_mask(bold_symlink_f)
            #   masked_data = apply_mask(bold_symlink_f, mask_img)
            #   k = k+1
            #   filename = datapath / f"functional_flattened/sub-{i}/{k}.npy"
            #   #create file
            #   np.save(filename, masked_data)

            event_file = bold_f.parent / bold_f.name.replace('_bold.nii.gz', '_events.tsv')
            #events = np.loadtxt(event_file, delimiter='\t', dtype={'names':('onset','duration','condition','item','key','RT'), 'formats':(int,int,str,int,int,float)})
            events = np.loadtxt(event_file,dtype=str)[1:]

            this_scenarios = []
            for e in events:
                condition, item = e[2], e[3]
                if condition not in event_to_scenario:
                    info(f'Skipping event {e}: no scanario mapping.')
                    continue
                skind, stype = event_to_scenario[condition]
                found = [s for s in scenarios if s['item'] == item]
                if not found:
                    info(f'Skipping event {e}: no scenario with this item found.')
                    continue
                found = found[0]
                assert found['type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {e}."
                text = f"{found['background']} {found['action']} {found['outcome']} {found[skind]}"
                this_scenarios.append(text)
            events_f = datapath / f"functional_flattened/sub-{i}/labels-{k}.npy"
            np.save(events_f, np.array(this_scenarios))

def download_dataset():
    cmd = 'cd data; datalad install --get-data https://github.com/OpenNeuroDatasets/ds000212.git'
    cprocess = run(cmd, shell=True)
    assert cprocess.returncode==0, f'"`{cmd}` failed: {cprocess}'

def info(msg):
    print(msg)

event_to_scenario = {
    "A_PHA": ("accidental", "Physical harm"),
    "B_PSA": ("accidental", "Psychological harm"),
    "C_IA": ("accidental", "Incest"),
    "D_PA": ("accidental", "Pathogen"),
    "E_NA": ("accidental", "Neutral"),
    "F_PHI": ("intentional", "Physical harm"),
    "G_PSI": ("intentional", "Psychological harm"),
    "H_II": ("intentional", "Incest"),
    "I_PI": ("intentional", "Pathogen"),
    "J_NI": ("intentional", "Neutral"),
}

def main():
    download_dataset()

    apply_atlas()

    generate_flattened_data()

    #load out of the numpy files as so
    test = np.load(datapath / 'functional_flattened/sub-03/1.npy')
    print(test)

if __name__ == '__main__':
    assert datapath.exists(), "expect this run in base directory with data/"
    main()
