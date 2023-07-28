import os
from typing import Union

import numpy as np
import torch as torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from pathlib import Path
from torch.nn import functional as F
from os import environ
from datetime import datetime

from utils.DS000212_LFB_Dataset import DS000212_LFB_Dataset
from utils.DS000212RawDataSet import DS000212RawDataset
# from DS000212_LFB_Dataset import DS000212_LFB_Dataset
# from DS000212RawDataSet import DS000212RawDataset

# returns a pandas dataframe of the CM training set (excluding long ones)
def load_ethics_ds(datapath: os.PathLike,
                   tokenizer: PreTrainedTokenizer,
                   config,
                   is_train=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Load csv:
    path = datapath / 'ethics/commonsense' / ('cm_train.csv' if is_train else 'cm_test.csv' )
    num_samples = config['num_samples_train'] if is_train else config['num_samples_test']
    df = pd.read_csv(os.path.abspath(path))
    df = df.drop(df[df.is_short == False].index)
    inputs, labels = df['input'].tolist()[:num_samples], df['label'][:num_samples]
    head_dims = len(set((df['label'])))
    # Tokenize:
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)
    tokens = torch.tensor(tokenized['input_ids'])  # shape: (num_samples, max_seq_len)
    masks = torch.tensor(tokenized['attention_mask'])  # shape: (num_samples, max_seq_len)
    target_tensors = torch.tensor(labels.tolist())  # shape: (num_samples, 1)
    # Create DataLoader
    data = TensorDataset(tokens, masks, target_tensors)
    dataloader = DataLoader(
        data, 
        batch_size=config['batch_size'], 
        shuffle=(config['shuffle_train'] if is_train else config['shuffle_test'])
    )
    return dataloader,head_dims

def load_ds000212_raw(datapath: os.PathLike,
                   tokenizer: PreTrainedTokenizer,
                   config,
                   subject=None):
    #ds000212 = DS000212RawDataset(
    #    datapath / 'ds000212_raw',
    #    datapath / 'ds000212_scenarios.csv',
    #    tokenizer
    #)
    ds000212 = DS000212_LFB_Dataset(
        datapath / 'ds000212_learning-from-brains',
        datapath / 'ds000212_scenarios.csv',
        tokenizer,
        subject=subject,
    )
    data_loader = DataLoader(
        ds000212,
        batch_size=config['batch_size'], 
        shuffle=False #shuffle=config['shuffle_train']
    )
    return data_loader, ds000212.target_head_dim

def load_ds000212(datapath: os.PathLike,
                   tokenizer: PreTrainedTokenizer,
                   config,
                   subject=None,
                   intervals=(-1,)):
    ds000212 = DS000212_LFB_Dataset(
        datapath / 'ds000212_learning-from-brains',
        datapath / 'ds000212_scenarios.csv',
        tokenizer,
        subject=subject,
        intervals=intervals
    )
    data_loader = DataLoader(
        ds000212,
        batch_size=config['batch_size']
    )
    return data_loader, ds000212.target_head_dim


def multiple_dataset_loading(datapath : Path, tokenizer, config) \
        -> tuple[list[DataLoader], list[Union[int, tuple[int, int]]]]:
    train_head_dims = []
    dataloaders = []
    ds_to_loader_table = {
        'ds000212_raw': load_ds000212_raw,
        'ds000212': load_ds000212,
        'ethics': load_ethics_ds
    }
    for train_dataset in config['train_datasets']:
        if not train_dataset in ds_to_loader_table: 
            raise Exception(f'Can not load dataset "{train_dataset}": no loader found. Loaders: \n{ds_to_loader_table}.')
        loader = ds_to_loader_table[train_dataset]
        dataloader, head_dims = loader( datapath, tokenizer, config)
        dataloaders.append(dataloader)
        train_head_dims.append(head_dims)
    return dataloaders, train_head_dims

def return_path_to_latest_checkpoint() -> Path:
    """
    Iterates through the artifacts folder to find the latest saved checkpoint for calculation of brain scores
    :return: path to checkpoint
    """
    artifactspath = Path(environ.get('AISCBB_ARTIFACTS_DIR', '/artifacts'))
    subdirectories = [d for d in os.listdir(artifactspath) if os.path.isdir(os.path.join(artifactspath, d))]
    if not subdirectories:
        return None

    #sort subdirectories to get most recent directory first
    #technically, checkpoint names aren't fully accurate to your specific timezone necessarily. but they're consistent
    subdirectories_sorted = []
    for i in subdirectories:
        converted = datetime.strptime(i, '%y%m%d-%H%M%S')
        subdirectories_sorted.append((converted, i))

    subdirectories_sorted_final = sorted(subdirectories_sorted, key=lambda x: x[0], reverse=True)

    for i in subdirectories_sorted_final:

        #checkpoints path
        checkpoint_path = Path(i[1] + '/version_0/checkpoints')
        checkpoint_path = os.path.join(artifactspath, checkpoint_path)

        if os.path.isdir(checkpoint_path):
            items_in_directory = os.listdir(checkpoint_path)
            if len(items_in_directory) == 1:
                # Get the path to the single item in the directory
                item_name = items_in_directory[0]
                item_path = os.path.join(checkpoint_path, item_name)
                return item_path
            else:
                #error, multiple checkpoints from one run.
                print("Error in return path to latest checkpoint, multiple checkpoints in run ", i[1])

# path = return_path_to_latest_checkpoint()
# print(path)