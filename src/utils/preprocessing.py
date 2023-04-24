import torch

from typing import Any
from transformers import PreTrainedTokenizer
from torch.utils.data import TensorDataset, DataLoader


def preprocess(tokens: torch.tensor,
               masks: torch.tensor,
               target: torch.tensor,
               config) -> tuple[DataLoader, DataLoader]:
    """
    Given the tokens, masks, and targets, it returns a dataloader with the tokens, masks, and targets.
    """
    data = TensorDataset(tokens, masks, target)
    data_train, data_val = torch.utils.data.random_split(
        data,
        [int(config['fraction_train'] * len(data)),
         len(data) - int(config['fraction_train'] * len(data))]
    )
    data_loader_train = DataLoader(data_train,
                                   config['batch_size'],
                                   config['shuffle_train'])
    data_loader_val = DataLoader(data_val,
                                 config['batch_size'],
                                 config['shuffle_train'])
    return data_loader_train, data_loader_val


def preprocess_prediction(inputs: list[str],
                          tokenizer: PreTrainedTokenizer,
                          batch_size: int = 1) -> DataLoader:
    """
    Given a list of strings, it tokenizes the strings and returns a dataloader with only the tokens and masks and
    no targets.
    """
    # tokenize (and truncate just in case)
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)

    # convert tokens, masks, and targets into tensors
    tokens = torch.tensor(tokenized['input_ids'])
    masks = torch.tensor(tokenized['attention_mask'])
    data = TensorDataset(tokens, masks)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader
