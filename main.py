import os
import math
from typing import Union, Literal, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModel
from einops import rearrange
import pandas as pd

training_type = 'regression' # can be 'classification' or 'regression'
regression_out_dims = (4, 20)
only_train_head = False 

# hyperparams
num_epochs = 10
batches_per_epoch = 100
batch_size = 4
checkpoint = 'bert-base-cased' # Hugging Face model we'll be using

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model including the base and the head
# note that out_dim can be either an int (for classification) or
# a tuple of ints (for regression)
class BERT(nn.Module):
    def __init__(self, base_model, out_dim: Union[int, tuple[int]] = 2):
        super().__init__()
        self.base = base_model
        # if the desired output has multiple axes, we want to output it flattened
        # and then reshape it at the end
        self.in_dim = 512*768 # in_dim = seq_len * d_model
        self.out_dim = out_dim
        self.out_dim_is_multidim = type(out_dim) is tuple # bool
        out_dim_flat = math.prod(out_dim) if self.out_dim_is_multidim else out_dim
        self.head = nn.Linear(self.in_dim, out_dim_flat) 

    def forward(self, tokens, mask):
        out = self.base(tokens, mask).last_hidden_state # [batch seq_len d_model]
        out = rearrange(out, 'batch pos d_model -> batch (pos d_model)') # [batch d_hidden_flat]
        out = self.head(out) # [batch d_out_flat]

        # if out_dim is multi-dimensional, reshape the output
        if self.out_dim_is_multidim:
            d_batch = out.shape[0]
            out = out.reshape((d_batch, *self.out_dim)) # [batch *d_out]

        return out

# lightning wrapper for training (scaling, parallelization etc)
class LitBert(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            only_train_head: bool = False,
            loss_name: Literal['cross-entropy', 'mse'] = 'cross-entropy'
        ):
        super().__init__()
        self.model = model
        self.only_train_head = only_train_head
        self.loss_name = loss_name
    
    def training_step(self, batch, _):
        tokens, mask, target = batch
        out = self.model(tokens, mask) # output

        # compute loss
        if self.loss_name == 'cross-entropy':
            loss = F.cross_entropy(out, target)
        elif self.loss_name == 'mse':
            loss = F.mse_loss(out, target)
        else:
            print(f"\n\nUnsupported loss name {self.loss_name}\n")
        
        # log and return
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        if self.only_train_head:
            for param in model.base.parameters():
                param.requires_grad = False
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
    
# given a list of strings and targets, it tokenizes the strings and returns
# a tensor of targets
def preprocess(inputs: list[str], targets: Any):
    # tokenize
    # TODO handle truncation (we're cutting the long entries short)
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)

    # convert tokens, masks, and targets into tensors
    tokens = torch.tensor(tokenized['input_ids']).to(device)
    masks = torch.tensor(tokenized['attention_mask']).to(device)
    if type(targets) is not torch.Tensor:
        targets = torch.tensor(targets).to(device)
    return tokens, masks, targets

# returns a pandas dataframe of the CM training set
def load_cm_df() -> pd.DataFrame:
    return pd.read_csv(os.path.abspath('./ethics/commonsense/cm_train.csv'))

# this is mostly just a placeholder for testing
def load_cm_text() -> DataLoader:
    df = load_cm_df()

    # put into data loader
    data = TensorDataset(*preprocess(df['input'].tolist(), df['label']))
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

# returns a dataset with the same commonsense inputs but random tensor outputs
def load_regression_placeholder() -> DataLoader:
    df = load_cm_df()
    n_samples = df.shape[0]
    targets = torch.rand((n_samples, *regression_out_dims)).to(device)

    # put into data loader
    data = TensorDataset(*preprocess(df['input'].tolist(), targets))
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader


# inference from the model on text
def classifier(text: str) -> torch.Tensor:
    tokenized = tokenizer([text], padding='max_length', truncation=True)
    tokens = torch.tensor(tokenized['input_ids']).to(device)
    mask = torch.tensor(tokenized['attention_mask']).to(device)
    logits = model(tokens, mask)
    return F.softmax(logits, dim=-1)
    
if __name__ == '__main__':
    # determine best device to run on
    # TODO: right now this doesn't work - we need to put the base model on the device
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available()
    #     else 'mps' if torch.backends.mps.is_available() # acceleration on Mac
    #     else 'cpu')
    device = torch.device('cpu') # placeholder
    print(f"{device=}")

    if training_type == 'classification':
        train_loader = load_cm_text() # load the data
    else:
        train_loader = load_regression_placeholder()

    # define the models (base, base+head, lightning wrapper)
    out_dim = 2 if training_type == 'classification' else regression_out_dims
    loss_name = 'cross-entropy' if training_type == 'classification' else 'mse'
    base_model = AutoModel.from_pretrained(checkpoint).to(device)
    model = BERT(base_model, out_dim).to(device)
    lit_model = LitBert(model, only_train_head, loss_name)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
    )
    trainer.fit(lit_model, train_loader)

    # test the model
    print(classifier("I hid a weapon under my clothes so nobody would notice it."))
