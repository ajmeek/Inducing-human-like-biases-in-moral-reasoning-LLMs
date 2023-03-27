import os

import numpy as np
import torch
import pandas as pd
from transformers import PreTrainedTokenizer, AutoTokenizer


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

