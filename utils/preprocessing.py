import torch

from typing import Any
from transformers import PreTrainedTokenizer
from torch.utils.data import TensorDataset, DataLoader

# given a list of strings and targets, it tokenizes the strings and returns
# a tensor of targets
def preprocess(inputs: list[str], targets: list[Any], tokenizer: PreTrainedTokenizer, batch_size: int = 4) -> DataLoader:
    # tokenize (and truncate just in case)
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)

    # convert tokens, masks, and targets into tensors
    tokens = torch.tensor(tokenized['input_ids'])
    masks = torch.tensor(tokenized['attention_mask'])
    target_tensors = []
    for target in targets:
        if type(target) is not torch.Tensor:
            target = torch.tensor(target, dtype=torch.long)
        target_tensors.append(target)
    data = TensorDataset(tokens, masks, *target_tensors)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

def preprocess_test(inputs: list[str], targets: list[Any], tokenizer: PreTrainedTokenizer, num_samples, batch_size: int = 4) -> DataLoader:
    # tokenize (and truncate just in case)
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)

    # convert tokens, masks, and targets into tensors
    tokens = torch.tensor(tokenized['input_ids'])[:num_samples, :]
    masks = torch.tensor(tokenized['attention_mask'])[:num_samples, :]
    target_tensors = torch.tensor(targets[:num_samples].tolist())
    data = TensorDataset(tokens, masks, target_tensors)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return train_loader


def preprocess_prediction(inputs: list[str], tokenizer: PreTrainedTokenizer, batch_size: int = 1) -> DataLoader:
    # tokenize (and truncate just in case)
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)

    # convert tokens, masks, and targets into tensors
    tokens = torch.tensor(tokenized['input_ids'])
    masks = torch.tensor(tokenized['attention_mask'])
    data = TensorDataset(tokens, masks)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return test_loader
