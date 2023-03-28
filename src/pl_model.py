from typing import Literal, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT

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
        # predictions is a list of tensors, one tensor per head. Each tensor is [batch, *head_dims]
        predictions = self.model(tokens, mask)

        # Compute loss and sum the weighted loss of each head.
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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        tokens, mask, *targets = batch
        predictions = self.model(tokens, mask)
        logits = predictions[0]
        probs = F.softmax(logits, dim=-1)
        predicted_label, confidence = probs.argmax(dim=-1), probs.max(dim=-1)
        self.log("test_acc", (predicted_label == targets[0]).float().mean(), prog_bar=True)  # This automatically accumulates the accuracy over the whole test set.
        return predicted_label  # This automatically accumulates the accuracy over the whole test set.

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        tokens, mask = batch
        predictions = self.model(tokens, mask)
        logits = predictions[0]
        return F.softmax(logits, dim=-1)
    
    def configure_optimizers(self):
        #! FIXME this breaks if you first only train head and then train the whole thing
        if self.only_train_head:
            for param in self.model.base.parameters():
                param.requires_grad = False
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
