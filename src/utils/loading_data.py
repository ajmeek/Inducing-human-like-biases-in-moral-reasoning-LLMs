import os

import numpy as np
import torch
import pandas as pd
from transformers import PreTrainedTokenizer, AutoTokenizer
from pathlib import Path


# returns a pandas dataframe of the CM training set (excluding long ones)
def load_csv_to_tensors(path: str | os.PathLike,
                        tokenizer: PreTrainedTokenizer,
                        num_samples: int) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    df = pd.read_csv(os.path.abspath(path))
    df = df.drop(df[df.is_short == False].index)
    inputs, labels = df['input'].tolist()[:num_samples], df['label'][:num_samples]
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)
    tokens = torch.tensor(tokenized['input_ids'])
    masks = torch.tensor(tokenized['attention_mask'])
    target_tensors = torch.tensor(labels.tolist())
    return tokens, masks, [target_tensors]


def load_np_fmri_to_tensor(base_path: str,
                           tokenizer: PreTrainedTokenizer,
                           num_samples: int,
                           normalize=True) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Load the fmri data from the npy files and return the tokens, masks, and target tensors.
    The structure of the data is as follows: the base_path contains a directory for each subject.
    Each subject directory contains several files labeled labels-*.npy and *.npy.
    :param num_samples: the number of samples to load. Note that the current numpy matrix will always be loaded in full.
    So you might get slightly more samples than you asked for.
    """
    tokens_list = []
    masks_list = []
    target_tensors_list = []
    current_num_samples = 0
    for directory in os.listdir(base_path):
        for file in os.listdir(os.path.join(base_path, directory)):
            if file.endswith('.npy') and file.startswith('labels'):
                fmri = np.load(os.path.join(base_path, directory, file.replace('labels-', '')))
                text = np.load(os.path.join(base_path, directory, file)).tolist()
                tokenized = tokenizer(text, padding='max_length', truncation=True)
                tokens_list.append(torch.tensor(tokenized['input_ids']))
                masks_list.append(torch.tensor(tokenized['attention_mask']))
                target_tensors_list.append(torch.tensor(fmri))
                current_num_samples = current_num_samples + len(text)
            if current_num_samples >= num_samples:
                break

    tokens = torch.cat(tokens_list, dim=0)
    masks = torch.cat(masks_list, dim=0)
    target_tensors = torch.cat(target_tensors_list, dim=0)
    if normalize:
        target_tensors = (target_tensors - target_tensors.mean()) / target_tensors.std()
    return tokens, masks, [target_tensors]


def load_ds000212_dataset(
    datapath: Path

):
    assert datapath.exists()
    scenarios = []
    fmri_items = []
    for subject_dir in Path(datapath / 'functional_flattened').glob('sub-*'):
        for runpath in subject_dir.glob('[0-9]*.npy'):
            scenario_path = runpath.parent / f'labels-{runpath.name}'
            fmri_items += np.load(runpath.resolve()).tolist()
            scenarios += np.load(scenario_path.resolve()).tolist()
    assert len(scenarios) == len(fmri_items), f'Expected: {len(scenarios)} == {len(fmri_items)}'
    # Drop those of inconsistent len:
    from collections import Counter
    counts = Counter(len(e) for e in fmri_items)
    most_common_len = counts.most_common()[0][0]
    indeces = [i for i, e in enumerate(fmri_items) if len(e) == most_common_len] 
    scenarios = [e for i,e in enumerate(scenarios) if i in indeces]
    fmri_items = [e for i,e in enumerate(fmri_items) if i in indeces]
    return {'inputs': scenarios, 'outputs': t.tensor(fmri_items)}



