import torch

from typing import Any
from transformers import PreTrainedTokenizer
from torch.utils.data import TensorDataset, DataLoader


def preprocess(tokens: torch.tensor,
               masks: torch.tensor,
               target: torch.tensor,
               batch_size: int = 4,
               shuffle: bool = True) -> DataLoader:
    """
    Given the tokens, masks, and targets, it returns a dataloader with the tokens, masks, and targets.
    """
    data = TensorDataset(tokens, masks, target)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


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
