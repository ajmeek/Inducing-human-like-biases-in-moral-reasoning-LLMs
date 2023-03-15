import os
from typing import Literal, Any
import torch
import pandas as pd

# returns a pandas dataframe of the CM training set (excluding long ones)
def load_cm_df(mode: Literal['train', 'test'] = 'train') -> pd.DataFrame:
    df = pd.read_csv(os.path.abspath(f'./ethics/commonsense/cm_{mode}.csv'))
    df = df.drop(df[df.is_short == False].index)
    return df

# this is mostly just a placeholder for testing
def load_cm_text() -> tuple[list[str], list[Any]]:
    df = load_cm_df()
    return df['input'].tolist(), df['label']

# returns a dataset with the same commonsense inputs but random tensor outputs
def load_regression_placeholder(regression_out_dims: tuple[int]) -> tuple[list[str], list[Any]]:
    df = load_cm_df()
    n_samples = df.shape[0]
    targets = torch.rand((n_samples, *regression_out_dims))
    return df['input'].tolist(), targets

# returns a dataset with both classification targets and random regression targets
def load_cm_with_reg_placeholder(regression_out_dims: tuple[int]) -> tuple[list[str], list[Any]]:
    df = load_cm_df()
    n_samples = df.shape[0]
    inputs = df['input'].tolist()
    cls_target = df['label']
    reg_target = torch.rand((n_samples, *regression_out_dims))
    return inputs, [cls_target, reg_target]
