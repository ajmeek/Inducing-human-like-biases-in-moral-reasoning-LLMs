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
from subprocess import run
from csv import DictReader

datapath = Path('./data')

#loop to turn all functional mri data by subject into 2d arrays. This can then be flattened for use in BERT linear layer fine tuning
def generate_flattened_data():
    #download the dataset into the data folder, but don't add it to git
    fmri_path = datapath / "ds000212"

    scenarios = []
    with open(datapath / "scenarios.csv", newline='') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            scenarios.append(row)
    #scenarios = np.loadtxt(
    #    datapath / "scenarios.csv", 
    #    delimiter=',',
    #    quotechar='"', 
    #    dtype= [
    #        ('item', 'S20'),
    #        ('type','U9999'),
    #        ('background', 'U9999'),
    #        ('action','U9999'),
    #        ('outcome','U9999'),
    #        ('accidental','U9999'),
    #        ('intentional','U9999'),
    #    ]
    #) 

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
            k = k+1
            bold_symlink_f = str(bold_f.resolve())
            mask_img = compute_epi_mask(bold_symlink_f)
            masked_data = apply_mask(bold_symlink_f, mask_img)
            filename = datapath / f"functional_flattened/sub-{i}/{k}.npy"
            #create file
            np.save(filename, masked_data)

            event_file = bold_f.parent / bold_f.name.replace('_bold.nii.gz', '_events.tsv')
            #events = np.loadtxt(event_file, delimiter='\t', dtype={'names':('onset','duration','condition','item','key','RT'), 'formats':(int,int,str,int,int,float)})
            #events = np.loadtxt(event_file,dtype=str)[1:]

            this_scenarios = []
            with open(event_file, newline='') as csvfile:
                reader = DictReader(csvfile, delimiter='\t', quotechar='"')
                for event in reader:
                    condition, item = event['condition'], event['item']
                    if condition not in event_to_scenario:
                        info(f'Skipping event {event}: no scanario mapping.')
                        continue
                    skind, stype = event_to_scenario[condition]
                    found = [s for s in scenarios if s['item'] == item]
                    if not found:
                        info(f'Skipping event {event}: no scenario with this item found.')
                        continue
                    found = found[0]
                    assert found['type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {event}."
                    text = f"{found['background']} {found['action']} {found['outcome']} {found[skind]}"
                    this_scenarios.append(text)
            print(this_scenarios)
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
    generate_flattened_data()

    #load out of the numpy files as so
    test = np.load(datapath / 'functional_flattened/sub-03/1.npy')
    print(test)

if __name__ == '__main__':
    assert datapath.exists(), "expect this run in base directory with data/"
    main()
