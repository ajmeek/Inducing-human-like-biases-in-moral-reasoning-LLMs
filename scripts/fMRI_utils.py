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
    scenarios = np.loadtxt(datapath / "scenarios.csv", delimiter=',', quotechar='"', dtype=str) 

    #this data structure will allow us to capture anatomical or functional data dependent on run and subject
    layout = bids.BIDSLayout(fmri_path, config=['bids', 'derivatives'])
    for i in list(layout.get_subjects()):
        #print(layout.get(subject=i,extension='nii.gz'))
        subject_path = datapath / f"ds000212/sub-{i}/func/"
        target_path = datapath / f"functional_flattened/sub-{i}"
        target_path.mkdir(parents=True, exist_ok=True)
        k = 0
        for bold_f in subject_path.glob("*.nii.gz"):
            mask_img = compute_epi_mask(bold_f)
            masked_data = apply_mask(bold_f, mask_img)
            k = k+1
            filename = datapath / f"functional_flattened/sub-{i}/{k}.npy"
            #create file
            np.save(filename, masked_data)

            event_file = bold_f.parent / bold_f.name.replace('_bold.nii.gz', '_events.tsv')
            events = np.loadtxt(event_file, delimiter='\t', dtype=str)
            this_scenarios = []
            for e in events:
                skind, stype = event_to_scenario[e['condition']]
                item = e['item']
                found = next(s for s in scenarios if s['item'] == item)
                assert found['type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {e}."
                text = f"{found['background']} {found['action']} {found['outcome']} {found[skind]}"
                this_scenarios.append(text)
            events_f = datapath / f"functional_flattened/sub-{i}/labels-{k}.npy"
            np.save(events_f, np.array(this_scenarios))


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
    generate_flattened_data()

    #load out of the numpy files as so
    test = np.load(datapath / 'functional_flattened/sub-03/1.npy')
    print(test)

if __name__ == '__main__':
    assert datapath.exists(), "expect this run in base directory with data/"
    main()
