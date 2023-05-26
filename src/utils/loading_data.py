import os
from typing import Union

import numpy as np
import torch as torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from pathlib import Path
from torch.nn import functional as F

from utils.DS000212RawDataSet import DS000212RawDataset


# returns a pandas dataframe of the CM training set (excluding long ones)
def load_csv_to_tensors(path: os.PathLike,
                        tokenizer: PreTrainedTokenizer,
                        num_samples: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(os.path.abspath(path))
    df = df.drop(df[df.is_short == False].index)
    inputs, labels = df['input'].tolist()[:num_samples], df['label'][
                                                         :num_samples]
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)
    tokens = torch.tensor(
        tokenized['input_ids'])  # shape: (num_samples, max_seq_len)
    masks = torch.tensor(
        tokenized['attention_mask'])  # shape: (num_samples, max_seq_len)
    target_tensors = torch.tensor(labels.tolist())  # shape: (num_samples, 1)
    # Create DataLoader
    data = TensorDataset(tokens, masks, target_tensors)
    dataloader = DataLoader(
        data, 
        batch_size=config['batch_size'], 
        shuffle=(config['shuffle_train'] if is_train else config['shuffle_test'])
    )
    return dataloader,head_dims

def load_ds000212(datapath: os.PathLike,
                   tokenizer: PreTrainedTokenizer,
                   config):
    ds000212 = DS000212RawDataset(
        datapath / 'ds000212_raw',
        datapath / 'ds000212_scenarios.csv',
        tokenizer
    )
    data_loader = DataLoader(
        ds000212,
        batch_size=config['batch_size'], 
        shuffle=config['shuffle_train']
    )
    return data_loader, ds000212.target_head_dim

def multiple_dataset_loading(datapath : Path, tokenizer, config) \
        -> tuple[list[DataLoader], list[Union[int, tuple[int, int]]]]:
    train_head_dims = []
    dataloaders = []
    ds_to_loader_table = {
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
