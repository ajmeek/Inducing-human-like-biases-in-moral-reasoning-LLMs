import os
import torch
import pandas as pd
from transformers import PreTrainedTokenizer


# returns a pandas dataframe of the CM training set (excluding long ones)
def load_csv_to_tensors(path: str,
                        tokenizer: PreTrainedTokenizer,
                        num_samples: int) -> tuple[torch.Tensor,
                                                   torch.Tensor,
                                                   list[torch.Tensor]]:
    df = pd.read_csv(os.path.abspath(path))
    df = df.drop(df[df.is_short == False].index)
    inputs, labels = df['input'].tolist()[:num_samples], df['label'][:num_samples]
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)
    tokens = torch.tensor(tokenized['input_ids'])
    masks = torch.tensor(tokenized['attention_mask'])
    target_tensors = torch.tensor(labels.tolist())
    return tokens, masks, [target_tensors]
