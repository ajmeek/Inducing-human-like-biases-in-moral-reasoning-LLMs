import os
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer
import torch
import pandas as pd

class EthicsDataset(TensorDataset):
    def __init__(self, 
                 context, 
                 tokenizer: PreTrainedTokenizer, 
                 is_train=True
        ) -> None:
        datapath = context['datapath']
        # Load csv:
        # path = datapath / 'ethics/commonsense' / ('cm_train.csv' if is_train else 'cm_test.csv' ) 
        path =  datapath + '/' + 'ethics/commonsense' + '/' + ('cm_train.csv' if is_train else 'cm_test.csv' )
        num_samples = context['num_samples_train'] if is_train else context['num_samples_test']
        df = pd.read_csv(os.path.abspath(path))
        df = df.drop(df[df.is_short == False].index)
        inputs, labels = df['input'].tolist()[:num_samples], df['label'][:num_samples]
        self.head_dims = len(set((df['label'])))
        # Tokenize:
        tokenized = tokenizer(inputs, padding=True, truncation=True) # padding='max_length'
        tokens = torch.tensor(tokenized['input_ids'])  # shape: (num_samples, max_seq_len)
        masks = torch.tensor(tokenized['attention_mask'])  # shape: (num_samples, max_seq_len)
        target_tensors = torch.tensor(labels.tolist())  # shape: (num_samples, 1)
        super().__init__(tokens, masks, target_tensors)
