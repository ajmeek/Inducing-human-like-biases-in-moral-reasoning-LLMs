import os
from typing import Union

import numpy as np
import torch as torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
from pathlib import Path
from torch.nn import functional as F

from src.utils.preprocessing import preprocess


# returns a pandas dataframe of the CM training set (excluding long ones)
def load_csv_to_tensors(path: os.PathLike,
                        tokenizer: PreTrainedTokenizer,
                        num_samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(os.path.abspath(path))
    df = df.drop(df[df.is_short == False].index)
    inputs, labels = df['input'].tolist()[:num_samples], df['label'][:num_samples]
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)
    tokens = torch.tensor(tokenized['input_ids'])  # shape: (num_samples, max_seq_len)
    masks = torch.tensor(tokenized['attention_mask'])  # shape: (num_samples, max_seq_len)
    target_tensors = torch.tensor(labels.tolist())  # shape: (num_samples, 1)
    return tokens, masks, target_tensors


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
    datapath: Path,
    tokenizer: PreTrainedTokenizer,
    num_samples: int,
    normalize=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor
]:
    print('Loading ds000212_dataset')
    assert datapath.exists()
    scenarios = []
    fmri_items = []
    for subject_dir in Path(datapath / 'functional_flattened').glob('sub-*'):
        for runpath in subject_dir.glob('[0-9]*.npy'):
            scenario_path = runpath.parent / f'labels-{runpath.name}'
            fmri_items += torch.tensor(np.load(runpath.resolve()))
            scenarios += np.load(scenario_path.resolve()).tolist()
    assert len(scenarios) == len(fmri_items), f'Expected: {len(scenarios)} == {len(fmri_items)}'
    # Pad with 0 those smaller one
    max_l = max(e.shape[0] for e in fmri_items)
    fmri_items = [F.pad(e, (0, max_l - e.shape[0]), 'constant', 0) for e in fmri_items]
    assert all(e.shape[0] == max_l for e in fmri_items)

    #from collections import Counter
    #counts = Counter(len(e) for e in fmri_items)
    #counts_mc = counts.most_common()  # TODO: Merge? 229 different len 
    #most_common_len = counts_mc[0][0]
    #indeces = [i for i, e in enumerate(fmri_items) if len(e) == most_common_len] 
    #scenarios = [e for i,e in enumerate(scenarios) if i in indeces]
    #fmri_items = [e for i,e in enumerate(fmri_items) if i in indeces]

    # Extract tokens, masks:
    tokens_list = []
    masks_list = []
    for text in scenarios:
        tokenized = tokenizer(text, padding='max_length', truncation=True)
        tokens_list.append(torch.tensor(tokenized['input_ids']))
        masks_list.append(torch.tensor(tokenized['attention_mask']))
    tokens = torch.stack(tokens_list)
    masks = torch.stack(masks_list)
    target_tensors = torch.stack(fmri_items)
    if normalize:
        target_tensors = (target_tensors - target_tensors.mean()) / target_tensors.std()
    
    assert tokens.shape[0] == masks.shape[0] == target_tensors.shape[0], f'{tokens.shape=} {masks.shape=} {target_tensors.shape=}'
    random_indices = torch.randperm(target_tensors.shape[0])[:num_samples]

    return tokens[random_indices], masks[random_indices], target_tensors[random_indices]


def multiple_dataset_loading(datapath, tokenizer, config, shuffle, normalize_fmri) \
        -> tuple[list[DataLoader], list[Union[int, tuple[int, int]]]]:
    train_head_dims = []
    dataloaders = []
    for index, train_dataset in enumerate(config['train_datasets']):
        if train_dataset == 'ds000212':
            tokens, masks, target = load_ds000212_dataset(datapath, tokenizer,
                                                           config['num_samples_train'], normalize=normalize_fmri)
            train_head_dims.append(target.shape[1])  # [64]  # Classification head and regression head, for example [2, (10, 4)]
            dataloader = preprocess(tokens, masks, target, config['batch_size'], shuffle=shuffle)
            dataloaders.append(dataloader)

        elif train_dataset.startswith('ethics'):
            tokens, masks, target = load_csv_to_tensors(datapath / train_dataset, tokenizer,
                                                         num_samples=config['num_samples_train'])
            train_head_dims.append(2)
            dataloader = preprocess(tokens, masks, target, config['batch_size'], shuffle=shuffle)
            dataloaders.append(dataloader)
    return dataloaders, train_head_dims
