#!/usr/bin/env python3

'''
fMRI_utils_fmriprep.py

Generates dataset for supervised learning with input text and related fMRI data.

Note - I'm making some changes to this to get it sorted for preprocessed data. Made a copy and changed the name so
the work that we did for raw & difumo data is still alright.
'''

import bids
from pathlib import Path
from nilearn.masking import compute_epi_mask, apply_mask
import nilearn
import nilearn.maskers
import nibabel as nib
import numpy as np
from subprocess import run, PIPE
from csv import DictReader
import os

datapath = Path('./data')
all_scenarios = []


# loop to turn all functional mri data by subject into 2d arrays. This can then be flattened for use in BERT linear layer fine tuning
def generate_flattened_data():
    # download the dataset into the data folder, but don't add it to git
    fmri_path = datapath / "ds000212"

    # for generating ROI analysis
    # difumo_atlas = nilearn.datasets.fetch_atlas_difumo(dimension=64)
    # difumo_masker = nilearn.maskers.NiftiMapsMasker(difumo_atlas['maps'], resampling_target='data', detrend=True).fit()

    # this data structure will allow us to capture anatomical or functional data dependent on run and subject
    layout = bids.BIDSLayout(fmri_path, config=['bids', 'derivatives'])
    for subject in list(layout.get_subjects()):
        process_subject(subject)
        # process_subject_roi(difumo_masker, subject)


def process_subject(subject):
    # TODO Change this so that it only runs on .nii.gz files for which we have valid symlinks, also for variable number of runs

    cwd = os.path.dirname(os.path.realpath(__file__))  # current working directory
    # cmd = 'echo hello'
    cmd = f'cd data/ds000212/derivatives/fmriprep/sub-{subject}/func/; find . -xtype l'
    result = run(cmd, cwd=cwd, stdout=PIPE, shell=True)
    # print(result.stdout) #checking this

    not_bytes = result.stdout.decode()
    list_of_invalid_symlinks_paths = not_bytes.split('\n')
    # print(list_of_invalid_symlinks_paths)
    list_of_invalid_symlinks_names = []
    for i in range(len(list_of_invalid_symlinks_paths)):
        temp = list_of_invalid_symlinks_paths[i]
        list_of_invalid_symlinks_names.append(temp[2:])
    # print(list_of_invalid_symlinks_names)
    # the above list contains the relative path to all invalid symlinks, not their names themselves. change

    subject_path = datapath / f"ds000212/derivatives/fmriprep/sub-{subject}/func/"
    target_path = datapath / f"fmriprep_functional_flattened/sub-{subject}"
    target_path.mkdir(parents=True, exist_ok=True)

    # grabbing the bold files and data masks.
    '''
    Logic as follows:
    Can't just grab every single BOLD file. Need to grab exactly the one we want and the brain mask for it.
    Fortunately, those will be the ones with valid symlinks. So use glob as below, then subtract all the invalid
    symlinks from it.
    '''

    nifti_symlinks = subject_path.glob("*.nii.gz")
    nifti_symlinks_basename = []
    for i in nifti_symlinks:
        # print(i)
        # print(os.path.basename(i))
        nifti_symlinks_basename.append(os.path.basename(i))
    set_of_invalid_symlinks = set(list_of_invalid_symlinks_names)
    # print(set_of_invalid_symlinks)
    valid_nifti_symlinks = [i for i in nifti_symlinks_basename if i not in set_of_invalid_symlinks]
    # print("valid: ", valid_nifti_symlinks)

    # Need a way to separate out the brain masks.
    # the most elegant way to do this would be to have a tuple of bold file and associated brain mask. Then can just iterate
    # over all those and capture the runs simultaneously

    bold_files = []
    brain_masks = []
    for file in valid_nifti_symlinks:
        if "bold" in file:
            bold_files.append(file)
        elif "brain" in file:
            brain_masks.append(file)
    # print("bold: ", bold_files)
    # print("brain: ", brain_masks)

    # now combine them based on their task and run
    tasks = ['fb', 'dis']
    nifti_tuples = []
    temp_brain = ""
    temp_bold = ""
    for i in tasks:
        for j in range(6):
            for k in bold_files:
                if f'task-{i}' in k and f'run-{j + 1}' in k:
                    temp_bold = k
            for k in brain_masks:
                if f'task-{i}' in k and f'run-{j + 1}' in k:  # should this be f'run-0{j+1}' ? No - that's only for raw data
                    temp_brain = k
            # print(temp_bold)
            # print(temp_brain)
            if temp_bold != "":
                nifti_tuples.append((temp_bold, temp_brain))
            temp_bold = ""
            temp_brain = ""

    # print("nifti_tuples: ")
    # for i in nifti_tuples:
    #    print(i)

    for tuple in nifti_tuples:
        full_path_bold = f"data/ds000212/derivatives/fmriprep/sub-{subject}/func/" + f'{tuple[0]}'
        full_path_brain = f"data/ds000212/derivatives/fmriprep/sub-{subject}/func/" + f'{tuple[1]}'
        masked_data = apply_mask(full_path_bold, full_path_brain)

        print("masked data computed for ", tuple[0])
        print("shape of masked data: ", masked_data.shape)

        scenarios = extract_scenarios(subject, tuple[0])
        print(scenarios)

        # need to change all that code first.

        if masked_data.shape[0] > 0 and len(scenarios) > 0:
            merged = merge_fmri_and_scenarios(masked_data, scenarios)
            if merged:
                data, scenarios = merged
                filename = datapath / f"fmriprep_functional_flattened/sub-{subject}/{tuple[0]}.npy"
                np.save(filename, data)
                events_f = datapath / f"fmriprep_functional_flattened/sub-{subject}/labels-{tuple[0]}.npy"
                np.save(events_f, scenarios)

        pass


def merge_fmri_and_scenarios(data, scenarios):
    '''
    Merges fMRI data with text (scenarios). Masked data in format (time series, voxels).
    Ten stories were presented in each 5.5 min run; the total experiment, involving six runs,
    lasted 33.2 min. Rest blocks of 10 s were interleaved between each story.
    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4769633/
    '''

    # TODO - have this run for false belief tasks too. Notable differences - time of 136 seconds, etc.

    TIME_SERIES_NUM = 166
    SCENARIOS_NUM = 10
    SCENARIO_SEC = 22
    NUM_BEFORE_LAST = 1
    REST_SEC = 10
    HEMODYNAMIC_LAG = 8  # 8 seconds after onput of story, biggest BOLD response in brain
    # (6-8s generally, go with 8 for now)
    # assert data.shape[0] == TIME_SERIES_NUM, f"Expected fMRI time series number {TIME_SERIES_NUM} but {data.shape}"
    if not data.shape[0] == TIME_SERIES_NUM:
        info(f"Skipping. Expected fMRI time series number {TIME_SERIES_NUM} but {data.shape}")
        return
        # assert len(scenarios) == SCENARIOS_NUM, f"Expected {SCENARIOS_NUM} scenarios but {len(scenarios)}"
    if not data.shape[0] == TIME_SERIES_NUM:
        info(f"Expected {SCENARIOS_NUM} scenarios but {len(scenarios)}")
        return
    # Add story end time points:
    nums_filter = [SCENARIO_SEC]
    for i in range(SCENARIOS_NUM - 1):
        nums_filter += [nums_filter[-1] + REST_SEC + SCENARIO_SEC + HEMODYNAMIC_LAG]
    # Convert to fMRI time series numbers (taken every 2 sec) (and correct if needed):
    nums_filter = [e // 2 - NUM_BEFORE_LAST for e in nums_filter]
    filtered = data[nums_filter, :]
    assert filtered.shape[0] == len(scenarios), f"Expected {filtered.shape[0]=} == {len(scenarios)=}"
    return (filtered, scenarios)


def extract_fmri_data(subject, run_num, bold_f):
    bold_symlink_f = str(bold_f.resolve())
    mask_img = compute_epi_mask(bold_symlink_f)
    return apply_mask(bold_symlink_f, mask_img)


def process_subject_roi(masker, subject):
    # Note - this is still computed on the raw data for now.

    subject_path = datapath / f"ds000212/sub-{subject}/func/"
    target_path = datapath / f"ds000212_difumo/sub-{subject}"
    target_path.mkdir(parents=True, exist_ok=True)
    run_num = 0
    for bold_f in subject_path.glob("*.nii.gz"):
        run_num = run_num + 1
        info(f'Processing {bold_f.name}')
        masked_data = extract_fmri_data(subject, run_num, bold_f)

        # data = nib.load(bold_f)  # no errors from this !
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


def extract_scenarios(subject, bold_f):
    """
    Changes to this are as follows:
        The tsv file containing the events doesn't occur in the derivatives folder afaik.
        So going up and into the raw data by relative path, which is annoying.


    :param subject:
    :param bold_f:
    :return:
    """

    # extract task and run number from file name.
    tasks = ['fb', 'dis']
    task = ""
    for i in tasks:
        if f'task-{i}' in bold_f:
            task = i
    run = ""
    for j in range(6):
        if f'run-{j + 1}' in bold_f:
            run = j + 1

    event_file = datapath / f'ds000212/sub-{subject}/func/sub-{subject}_task-{task}_run-0{run}_events.tsv'
    print(event_file)

    # TODO - false belief event tsv files have different conditions. to properly label, amend mapping

    scenarios = []
    with open(event_file, newline='') as csvfile:
        reader = DictReader(csvfile, delimiter='\t', quotechar='"')
        # print(reader)
        # print("keys: ", reader.fieldnames)
        for event in reader:
            # print(event)
            condition, item = event['condition'], event['item']
            if condition not in event_to_scenario:
                info(f'Skipping event {event}: no scenario mapping.')
                continue
            skind, stype = event_to_scenario[condition]
            found = [s for s in all_scenarios if s['item'] == item]
            if not found:
                info(f'Skipping event {event}: no scenario with this item found.')
                continue
            found = found[0]
            assert found[
                       'type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {event}."
            text = f"{found['background']} {found['action']} {found['outcome']} {found[skind]}"
            scenarios.append(text)
    return scenarios


def init_scenarios():
    global all_scenarios
    with open(datapath / "scenarios.csv", newline='', encoding='utf-8') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            all_scenarios.append(row)


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
    # download_dataset() - use the script for this now
    init_scenarios()
    generate_flattened_data()
    # load out of the numpy files as so
    # test = np.load(datapath / 'functional_flattened/sub-03/1.npy')
    # print(test)
    # process_subject(subject="03")


if __name__ == '__main__':
    assert datapath.exists(), "expect this run in base directory with data/"
    main()
