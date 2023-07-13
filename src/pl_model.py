from typing import Literal, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
import copy

import lightning.pytorch as pl


# lightning wrapper for training (scaling, parallelization etc)
class LitBert(pl.LightningModule):
    def __init__(self, model: nn.Module, config : dict):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_names = config['loss_names']
        self.loss_weights = config['loss_weights'] or [1 for _ in self.loss_names]
        self.regularize_from_init = config['regularize_from_init']
        self.regularization_coef = config['regularization_coef']
        # store the initial weights of the model (used for regularization)
        # note that we're not applying the regularization to the heads
        self.init_params = copy.deepcopy([p for p in model.base.parameters()])
        self.dataset_names = config['train_datasets']

    def training_step(self, batch, _):
        loss = 0
        # Batch has now multiple dataloaders (can also still be 1)
        for index, dataloader in enumerate(batch):
            tokens, mask, target = dataloader
            predictions = self.model(tokens, mask)[index]

            # Compute weighted and summed loss
            loss_weight = self.loss_weights[index]
            loss_name = self.loss_names[index]
            if loss_name == 'cross-entropy':
                loss += loss_weight * F.cross_entropy(predictions, target)
            elif loss_name == 'mse':
                loss += loss_weight * F.mse_loss(predictions, target)
            else:
                print(f"\n\nUnsupported loss name {loss_name}\n")

        if self.regularize_from_init:
            # add regularization from the initial weights
            # (encourages staying closer to the pretrained base model weights)
            reg_loss = 0
            params = self.model.base.parameters()
            for w, w0 in zip(params, self.init_params):
                w0 = w0.to(w.device)
                reg_loss += torch.pow(w - w0, 2).sum()
            loss += self.regularization_coef * reg_loss

        # log and return
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        dataset_name = self.dataset_names[dataloader_idx]
        if dataset_name.startswith('ethics'):
            return self.test_step(batch, batch_idx, dataloader_idx)
        elif dataset_name == 'ds000212':
            tokens, mask, target = batch
            predictions = self.model(tokens, mask)[dataloader_idx]
            mse_loss = F.mse_loss(predictions, target)
            self.log("val_mse", mse_loss, prog_bar=True)
            return mse_loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        tokens, mask, target = batch
        predictions = self.model(tokens, mask)
        logits = predictions[0]  # Note: take the first, so we use the ETHICS head to predict.

        probs = F.softmax(logits, dim=-1)
        predicted_label = probs.argmax(dim=-1)
        # log the accuracy (this automatically accumulates it over the whole test set)
        self.log("test_acc", (predicted_label == target).float().mean(), prog_bar=True, sync_dist=True)
        return predicted_label

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        tokens, mask, target, dataset_index = batch
        predictions = self.model(tokens, mask)
        logits = predictions[dataset_index[0].item()]
        return F.softmax(logits, dim=-1)

    def configure_optimizers(self):
        # ! FIXME this breaks if you first only train head and then train the whole thing
        if self.config['only_train_head']:
            for param in self.model.base.parameters():
                param.requires_grad = False
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['lr']
        )
        return optimizer
