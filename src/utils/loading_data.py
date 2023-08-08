from typing import Union

import torch as torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

from utils.DS000212RawDataSet import DS000212RawDataset
from utils.DS000212_LFB_Dataset import DS000212_LFB_Dataset
from utils.EthicsDataset import EthicsDataset

DATASETS = {
    k.__name__ : k for k in (
        DS000212_LFB_Dataset,
        DS000212RawDataset,
        EthicsDataset
    )
}
DEFAULT_DATASETS = [
    DS000212_LFB_Dataset.__name__,
    EthicsDataset.__name__
]

def multiple_dataset_loading(tokenizer, context) \
        -> tuple[list[DataLoader], list[Union[int, tuple[int, int]]]]:
    train_head_dims = []
    dataloaders = []

    for ds_key in context['train_datasets']:
        ds_class = DATASETS[ds_key]
        ds = ds_class(context, tokenizer)
        dataloader = DataLoader(
            ds, 
            batch_size=context['batch_size'], 
            shuffle=(context['shuffle_train'] if not isinstance(ds, IterableDataset) else False)
        )
        head_dims = ds.head_dims
        dataloaders.append(dataloader)
        train_head_dims.append(head_dims)
    return dataloaders, train_head_dims
