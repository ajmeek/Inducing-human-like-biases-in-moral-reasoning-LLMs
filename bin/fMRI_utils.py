#!/usr/bin/env python3

'''
fMRI_utils.py
Generates dataset for supervised learning with input text and related fMRI data.
'''

import bids
from pathlib import Path
from nilearn.masking import compute_epi_mask, apply_mask
import nilearn
import nilearn.maskers
import nibabel as nib
import numpy as np
from subprocess import run
from csv import DictReader

datapath = Path('./data')
all_scenarios = []

#loop to turn all functional mri data by subject into 2d arrays. This can then be flattened for use in BERT linear layer fine tuning
def generate_flattened_data():
    #download the dataset into the data folder, but don't add it to git
    fmri_path = datapath / "ds000212"

    #for generating ROI analysis
    difumo_atlas = nilearn.datasets.fetch_atlas_difumo(dimension=64)
    difumo_masker = nilearn.maskers.NiftiMapsMasker(difumo_atlas['maps'], resampling_target='data', detrend=True).fit()

    #this data structure will allow us to capture anatomical or functional data dependent on run and subject
    layout = bids.BIDSLayout(fmri_path, config=['bids', 'derivatives'])
    for subject in list(layout.get_subjects()):
        process_subject(subject)
        process_subject_roi(difumo_masker, subject)

def process_subject(subject):
    subject_path = datapath / f"ds000212/sub-{subject}/func/"
    target_path = datapath / f"functional_flattened/sub-{subject}"
    target_path.mkdir(parents=True, exist_ok=True)
    run_num = 0
    for bold_f in subject_path.glob("*.nii.gz"):
        run_num = run_num+1
        info(f'Processing {bold_f.name}')
        masked_data = extract_fmri_data(subject, run_num, bold_f)
        scenarios = extract_scenarios(subject, run_num, bold_f)
        if masked_data.shape[0] > 0 and len(scenarios) > 0:
            merged = merge_fmri_and_scenarios(masked_data, scenarios)
            if merged:
                data, scenarios = merged
                filename = datapath / f"functional_flattened/sub-{subject}/{run_num}.npy"
                np.save(filename, data)
                events_f = datapath / f"functional_flattened/sub-{subject}/labels-{run_num}.npy"
                np.save(events_f, scenarios)

def merge_fmri_and_scenarios(data, scenarios):
    '''
    Merges fMRI data with text (scenarios). Masked data in format (time series, voxels).
    Ten stories were presented in each 5.5 min run; the total experiment, involving six runs,
    lasted 33.2 min. Rest blocks of 10 s were interleaved between each story.
    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4769633/
    '''
    TIME_SERIES_NUM=166
    SCENARIOS_NUM=10
    SCENARIO_SEC=22
    NUM_BEFORE_LAST=1
    REST_SEC=10
    HEMODYNAMIC_LAG = 8 #8 seconds after onput of story, biggest BOLD response in brain
                        # (6-8s generally, go with 8 for now)
    #assert data.shape[0] == TIME_SERIES_NUM, f"Expected fMRI time series number {TIME_SERIES_NUM} but {data.shape}"
    if not data.shape[0] == TIME_SERIES_NUM:
        info(f"Skipping. Expected fMRI time series number {TIME_SERIES_NUM} but {data.shape}")
        return
    # assert len(scenarios) == SCENARIOS_NUM, f"Expected {SCENARIOS_NUM} scenarios but {len(scenarios)}"
    if not data.shape[0] == TIME_SERIES_NUM:
        info(f"Expected {SCENARIOS_NUM} scenarios but {len(scenarios)}")
        return
    # Add story end time points:
    nums_filter=[SCENARIO_SEC]
    for i in range(SCENARIOS_NUM-1):
        nums_filter += [nums_filter[-1] + REST_SEC + SCENARIO_SEC + HEMODYNAMIC_LAG]
    # Convert to fMRI time series numbers (taken every 2 sec) (and correct if needed):
    nums_filter = [e//2 - NUM_BEFORE_LAST for e in nums_filter]
    filtered = data[nums_filter, :]
    assert filtered.shape[0] == len(scenarios), f"Expected {filtered.shape[0]=} == {len(scenarios)=}"
    return (filtered, scenarios)

def extract_fmri_data(subject, run_num, bold_f):
    bold_symlink_f = str(bold_f.resolve())
    mask_img = compute_epi_mask(bold_symlink_f)
    return apply_mask(bold_symlink_f, mask_img)

def process_subject_roi(masker, subject):
    subject_path = datapath / f"ds000212/sub-{subject}/func/"
    target_path = datapath / f"ds000212_difumo/sub-{subject}"
    target_path.mkdir(parents=True, exist_ok=True)
    run_num = 0
    for bold_f in subject_path.glob("*.nii.gz"):
        run_num = run_num+1
        info(f'Processing {bold_f.name}')
        masked_data = extract_fmri_data(subject, run_num, bold_f)

        #data = nib.load(bold_f)  # no errors from this !
        roi_time_series = masker.transform(masked_data)

        scenarios = extract_scenarios(subject, run_num, bold_f)
        if roi_time_series.shape[0] > 0 and len(scenarios) > 0:
            merged = merge_fmri_and_scenarios(roi_time_series, scenarios)
            if merged:
                data, scenarios = merged
                filename = datapath / f"ds000212_difumo/sub-{subject}/{run_num}.npy"
                np.save(filename, data)
                events_f = datapath / f"ds000212_difumo/sub-{subject}/labels-{run_num}.npy"
                np.save(events_f, scenarios)

def extract_scenarios(subject, run_num, bold_f):
    event_file = bold_f.parent / bold_f.name.replace('_bold.nii.gz', '_events.tsv')
    scenarios = []
    with open(event_file, newline='') as csvfile:
        reader = DictReader(csvfile, delimiter='\t', quotechar='"')
        for event in reader:
            condition, item = event['condition'], event['item']
            if condition not in event_to_scenario:
                info(f'Skipping event {event}: no scanario mapping.')
                continue
            skind, stype = event_to_scenario[condition]
            found = [s for s in all_scenarios if s['item'] == item]
            if not found:
                info(f'Skipping event {event}: no scenario with this item found.')
                continue
            found = found[0]
            assert found['type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {event}."
            text = f"{found['background']} {found['action']} {found['outcome']} {found[skind]}"
            scenarios.append(text)
    return scenarios

def init_scenarios():
    global all_scenarios
    with open(datapath / "scenarios.csv", newline='', encoding='utf-8') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            all_scenarios.append(row)

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
    init_scenarios()
    generate_flattened_data()

    #load out of the numpy files as so
    # test = np.load(datapath / 'functional_flattened/sub-03/1.npy')
    # print(test)

if __name__ == '__main__':
    assert datapath.exists(), "expect this run in base directory with data/"
    main()