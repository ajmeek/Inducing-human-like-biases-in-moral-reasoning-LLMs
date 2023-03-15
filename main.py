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

regression_out_dims = (4, 20)
only_train_head = True 

# hyperparams
num_epochs = 1
batches_per_epoch = 100
batch_size = 4
checkpoint = 'bert-base-cased' # Hugging Face model we'll be using

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model including the base and multiple heads
# the heads are specified by the head_dims argument - the dimensionality of
# each had can be an int or a tuple of ints
class BERT(nn.Module):
    def __init__(self, base_model, head_dims: list[Union[int, tuple[int]]] = [2]):
        super().__init__()
        self.base = base_model

        # initialize all the heads
        # if the desired output has multiple axes, we want to output it flattened
        # and then reshape it at the end
        self.head_dims = head_dims
        head_in_dim = base_model.config.hidden_size

        heads = []
        for head_d in head_dims:
            head_d_flat = math.prod(head_d) if type(head_d) is tuple else head_d
            heads.append(nn.Linear(head_in_dim, head_d_flat))
        self.heads = nn.ModuleList(heads)

    def forward(self, tokens, mask):
        base_out = self.base(tokens, mask) # [batch seq_len d_model]
        base_out = base_out.last_hidden_state # use last layer activations
        base_out = base_out[:, 0, :] # only take the encoding of [CLS] -> [batch, d_model]

        outs = []
        for head, head_d in zip(self.heads, self.head_dims):
            head_out = head(base_out) # [batch d_out_flat]

            # if out_dim is multi-dimensional, reshape the output
            if type(head_d) is tuple:
                d_batch = head_out.shape[0]
                head_out = head_out.reshape((d_batch, *head_d)) # [batch *head_d]

            outs.append(head_out)

        return outs

# lightning wrapper for training (scaling, parallelization etc)
class LitBert(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            only_train_head: bool = False,
            loss_names: list[Literal['cross-entropy', 'mse']] = ['cross-entropy']
        ):
        super().__init__()
        self.model = model
        self.only_train_head = only_train_head
        self.loss_names = loss_names
    
    def training_step(self, batch, _):
        tokens, mask, *targets = batch
        predictions = self.model(tokens, mask) # outputs

        # compute loss
        loss = 0
        for pred, target, loss_name in zip(predictions, targets, self.loss_names):
            if loss_name == 'cross-entropy':
                loss += F.cross_entropy(pred, target)
            elif loss_name == 'mse':
                loss += F.mse_loss(pred, target)
            else:
                print(f"\n\nUnsupported loss name {loss_name}\n")
        
        # log and return
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        if self.only_train_head: #! FIXME
            for param in model.base.parameters():
                param.requires_grad = False
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
    
# given a list of strings and targets, it tokenizes the strings and returns
# a tensor of targets
def preprocess(inputs: list[str], targets: list[Any]) -> DataLoader:
    # tokenize (and truncate just in case)
    tokenized = tokenizer(inputs, padding='max_length', truncation=True)

    # convert tokens, masks, and targets into tensors
    tokens = torch.tensor(tokenized['input_ids']).to(device)
    masks = torch.tensor(tokenized['attention_mask']).to(device)
    target_tensors = []
    for target in targets:
        if type(target) is not torch.Tensor:
            target = torch.tensor(target, dtype=torch.long).to(device)
        target_tensors.append(target)
    data = TensorDataset(tokens, masks, *target_tensors)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

# returns a pandas dataframe of the CM training set (excluding long ones)
def load_cm_df(mode: Literal['train', 'test'] = 'train') -> pd.DataFrame:
    df = pd.read_csv(os.path.abspath(f'./ethics/commonsense/cm_{mode}.csv'))
    df = df.drop(df[df.is_short == False].index)
    return df

# this is mostly just a placeholder for testing
def load_cm_text() -> DataLoader:
    df = load_cm_df()
    return preprocess(df['input'].tolist(), df['label'])

# returns a dataset with the same commonsense inputs but random tensor outputs
def load_regression_placeholder() -> DataLoader:
    df = load_cm_df()
    n_samples = df.shape[0]
    targets = torch.rand((n_samples, *regression_out_dims)).to(device)
    return preprocess(df['input'].tolist(), targets)

# returns a dataset with both classification targets and random regression targets
def load_cm_with_reg_placeholder() -> DataLoader:
    df = load_cm_df()
    n_samples = df.shape[0]
    inputs = df['input'].tolist()
    cls_target = df['label']
    reg_target = torch.rand((n_samples, *regression_out_dims)).to(device)
    return preprocess(inputs, [cls_target, reg_target])

# wrapper for the whole training process (incl tokenization)
def train_model(
        model: nn.Module,
        inputs: list[str],
        targets: list[Any],
        only_train_head: bool = False,
        loss_names: list[Literal['cross-entropy', 'mse']] = ['cross-entropy'],
        batches_per_epoch: int = 100,
        num_epochs: int = 10,
    ):
    loader = preprocess(inputs, targets)
    lit_model = LitBert(model, only_train_head, loss_names)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
    )
    trainer.fit(lit_model, loader) 

# inference from the model on text
def classifier(text: str) -> torch.Tensor:
    tokenized = tokenizer([text], padding='max_length', truncation=True)
    tokens = torch.tensor(tokenized['input_ids']).to(device)
    mask = torch.tensor(tokenized['attention_mask']).to(device)
    logits, reg_pred = model(tokens, mask)
    return F.softmax(logits, dim=-1)

# load the testing set and see how well our model performs on it
def test_accuracy(max_samples=100, log_all=False):
    model.eval()
    df = load_cm_df('test')
    correct_results = 0
    total_results = 0
    for i, row in df.iterrows(): # TODO: add batching for higher efficiency
        if i > max_samples: break
        text = row['input']
        logits = classifier(text)
        prediction, confidence = logits.argmax().item(), logits.max().item()
        label = row['label']
        if log_all:
            print(f"\n{text=:<128} {prediction=:<4} ({confidence=:.4f}) {label=:<4}\n")
        if prediction == label: correct_results += 1
        total_results += 1
    print(f"\n\nAccuracy {correct_results/total_results:.4f}\n")
    
if __name__ == '__main__':
    # determine best device to run on
    # TODO: right now this doesn't work - we need to put the base model on the device
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available()
    #     else 'mps' if torch.backends.mps.is_available() # acceleration on Mac
    #     else 'cpu')
    device = torch.device('cpu') # placeholder
    print(f"{device=}")

    train_loader = load_cm_with_reg_placeholder() # load the data

    # define the models (base, base+head, lightning wrapper)
    base_model = AutoModel.from_pretrained(checkpoint).to(device)
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    model = BERT(base_model, head_dims=[2, regression_out_dims]).to(device)
    lit_model = LitBert(model, only_train_head, loss_names=['cross-entropy', 'mse'])

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
    )
    trainer.fit(lit_model, train_loader)

    # test the model
    test_accuracy()
