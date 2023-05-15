from typing import Literal, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
import copy

import lightning.pytorch as pl



# lightning wrapper for training (scaling, parallelization etc)
class LitBert(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            only_train_head: bool = False,
            loss_names: list[Literal['cross-entropy', 'mse']] = ['cross-entropy'],
            loss_weights: list[float] = None,
            regularize_from_init: bool = False,
            regularization_coef: float = 0.,
        ):
        super().__init__()
        self.model = model
        self.only_train_head = only_train_head
        self.loss_names = loss_names
        self.loss_weights = [1 for _ in loss_names] if loss_weights is None else loss_weights
        self.regularize_from_init = regularize_from_init
        self.regularization_coef = regularization_coef
        # store the initial weights of the model (used for regularization)
        # note that we're not applying the regularization to the heads
        self.init_params = copy.deepcopy([p for p in model.base.parameters()])
    
    def training_step(self, batch, batch_idx):
        loss = 0
        for index, dataloader in enumerate(batch):  # Batch has now multiple dataloaders (can also still be 1)
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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        tokens, mask, target = batch
        predictions = self.model(tokens, mask)
        logits = predictions[0]  # Note: take the first, so we use the ETHICS head to predict.

        probs = F.softmax(logits, dim=-1)
        predicted_label = probs.argmax(dim=-1)
        # log the accuracy (this automatically accumulates it over the whole test set)
        self.log("test_acc", (predicted_label == target).float().mean(), prog_bar=True)
        return predicted_label

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        tokens, mask, target, dataset_index = batch
        predictions = self.model(tokens, mask)
        logits = predictions[dataset_index[0].item()]
        return F.softmax(logits, dim=-1)
    
    def configure_optimizers(self):
        #! FIXME this breaks if you first only train head and then train the whole thing
        if self.only_train_head:
            for param in self.model.base.parameters():
                param.requires_grad = False
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
