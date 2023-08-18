from typing import Union

import torch as torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from pathlib import Path
from torch.nn import functional as F
from os import environ
from datetime import datetime
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
    EthicsDataset.__name__,
    DS000212_LFB_Dataset.__name__
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