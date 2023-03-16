from typing import Literal, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import PreTrainedTokenizer
import lightning.pytorch as pl

# lightning wrapper for training (scaling, parallelization etc)
class LitBert(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            only_train_head: bool = False,
            loss_names: list[Literal['cross-entropy', 'mse']] = ['cross-entropy'],
            loss_weights: list[float | int] = None,
        ):
        super().__init__()
        self.model = model
        self.only_train_head = only_train_head
        self.loss_names = loss_names
        self.loss_weights = [1 for _ in loss_names] if loss_weights is None else loss_weights
    
    def training_step(self, batch, _):
        tokens, mask, *targets = batch
        predictions = self.model(tokens, mask) # outputs

        # compute loss
        loss = 0
        for pred, target, loss_name, loss_weight in zip(predictions,
                                                        targets,
                                                        self.loss_names,
                                                        self.loss_weights):
            if loss_name == 'cross-entropy':
                loss += loss_weight * F.cross_entropy(pred, target)
            elif loss_name == 'mse':
                loss += loss_weight * F.mse_loss(pred, target)
            else:
                print(f"\n\nUnsupported loss name {loss_name}\n")
        
        # log and return
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        #! FIXME this breaks if you first only train head and then train the whole thing
        if self.only_train_head:
            for param in self.model.base.parameters():
                param.requires_grad = False
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
     
# given a list of strings and targets, it tokenizes the strings and returns
# a tensor of targets
def preprocess(inputs: list[str], targets: list[Any],
               tokenizer: PreTrainedTokenizer, batch_size: int = 4) -> DataLoader:
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

# wrapper for the whole training process (incl tokenization)
def train_model(
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        inputs: list[str],
        targets: list[Any],
        loss_names: list[Literal['cross-entropy', 'mse']] = ['cross-entropy'],
        loss_weights: list[float | int] = None,
        only_train_head: bool = False,
        num_epochs: int = 10,
        batches_per_epoch: int = 100,
        batch_size: int = 4,
        device: str = 'cpu',
        n_devices: int = 1,
    ):
    loader = preprocess(inputs, targets, tokenizer, batch_size)
    lit_model = LitBert(model, only_train_head, loss_names, loss_weights)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
        accelerator=device,
        devices=n_devices,
    )
    trainer.fit(lit_model, loader) 
