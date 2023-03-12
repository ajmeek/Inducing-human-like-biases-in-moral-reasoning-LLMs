import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import lightning.pytorch as pl
from typing import Literal

# hyperparams
num_epochs = 10
batches_per_epoch = 100
batch_size = 4
checkpoint = 'bert-base-cased' # Hugging Face model we'll be using

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model including the base and the head
# TODO: add support for multi-axis tensor outputs (eg out_dim=(64, 32))
class BERT(nn.Module):
    def __init__(self, base_model, out_dim=2):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(512*768, out_dim) # in_dim = seq_len * d_model

    def forward(self, tokens, mask):
        out = self.base(tokens, mask).last_hidden_state
        out = rearrange(out, 'batch pos d_model -> batch (pos d_model)')
        out = self.head(out)
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
    
# this is mostly just a placeholder for testing
def load_cm_text() -> DataLoader:
    df = pd.read_csv(os.path.abspath('./ethics/commonsense/cm_train.csv'))

    # tokenize
    # TODO handle truncation (we're cutting the long entries short)
    tokenized = tokenizer(df['input'].tolist(), padding='max_length', truncation=True)

    # split up into tokens, attn masks, and labels
    tokens = torch.tensor(tokenized['input_ids']).to(device)
    masks = torch.tensor(tokenized['attention_mask']).to(device)
    targets = torch.tensor(df['label']).to(device)

    # put into data loader
    data = TensorDataset(tokens, masks, targets)
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

    train_loader = load_cm_text() # load the data

    # define the models (base, base+head, lightning wrapper)
    base_model = AutoModel.from_pretrained(checkpoint).to(device)
    model = BERT(base_model, out_dim=2).to(device)
    lit_model = LitBert(model, only_train_head=True)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
    )
    trainer.fit(lit_model, train_loader)

    # test the model
    print(classifier("I hid a weapon under my clothes so nobody would notice it."))
