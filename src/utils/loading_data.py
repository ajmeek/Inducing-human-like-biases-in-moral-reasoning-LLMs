from typing import Union

import torch as torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from utils.DS000212RawDataSet import DS000212RawDataset
from utils.DS000212_LFB_Dataset import DS000212_LFB_Dataset
from utils.EthicsDataset import EthicsDataset


# returns a pandas dataframe of the CM training set (excluding long ones)
def load_ethics_ds(tokenizer: PreTrainedTokenizer,
                   context,
                   is_train=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = EthicsDataset(tokenizer, context, is_train)
    dataloader = DataLoader(
        data, 
        batch_size=context['batch_size'], 
        shuffle=(context['shuffle_train'] if is_train else context['shuffle_test'])
    )
    return dataloader, data.head_dims

def load_ds000212_raw(tokenizer: PreTrainedTokenizer, context):
    ds000212 = DS000212_LFB_Dataset(context, tokenizer)
    data_loader = DataLoader(
        ds000212,
        batch_size=context['batch_size'], 
        shuffle=False
    )
    return data_loader, ds000212.target_head_dim

def load_ds000212( tokenizer: PreTrainedTokenizer,
                   context,
                   subject=None):
    ds000212 = DS000212_LFB_Dataset(context, tokenizer, subject=subject)
    data_loader = DataLoader(
        ds000212,
        batch_size=context['batch_size']
    )
    return data_loader, ds000212.target_head_dim


def multiple_dataset_loading(tokenizer, context) \
        -> tuple[list[DataLoader], list[Union[int, tuple[int, int]]]]:
    train_head_dims = []
    dataloaders = []
    ds_to_loader_table = {
        'ds000212_raw': load_ds000212_raw,
        'ds000212': load_ds000212,
        'ethics': load_ethics_ds
    }
    for train_dataset in context['train_datasets']:
        if not train_dataset in ds_to_loader_table: 
            raise Exception(f'Can not load dataset "{train_dataset}": no loader found. Loaders: \n{ds_to_loader_table}.')
        loader = ds_to_loader_table[train_dataset]
        dataloader, head_dims = loader(tokenizer, context)
        dataloaders.append(dataloader)
        train_head_dims.append(head_dims)
    return dataloaders, train_head_dims
