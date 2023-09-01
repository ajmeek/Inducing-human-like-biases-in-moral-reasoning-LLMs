import os
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import pandas as pd
import torch

class EthicsDataset(Dataset):
    def __init__(self, 
                 context, 
                 tokenizer: PreTrainedTokenizer, 
                 is_train=True
        ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        datapath = context['datapath']
        path = datapath / 'ethics/commonsense' / ('cm_train.csv' if is_train else 'cm_test.csv' )
        self._df = pd.read_csv(os.path.abspath(path))
        self._df = self._df.drop(self._df[self._df.is_short == False].index)
        num_samples = context['num_samples_train'] if is_train else context['num_samples_test']
        if num_samples > 0:
            self._df = self._df.iloc[:num_samples]
        self.head_dims = len(set((self._df['label'])))

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        input_ , label = self._df.iloc[index][['input', 'label']]
        tokenized = self._tokenizer(input_, padding='max_length', truncation=True)
        return torch.tensor(tokenized['input_ids']), torch.tensor(tokenized['attention_mask']), torch.tensor(label)