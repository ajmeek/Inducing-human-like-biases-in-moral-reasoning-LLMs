from dataclasses import dataclass
from typing import Optional, Union
from utils.BrainBiasDataModule import DatasetConfig
from datasets import (
    Dataset,
    IterableDataset,
    ClassLabel,
    Value,
    ClassLabel,
    Sequence,
)
from utils.BrainBiasDataModule import BrainBiasDataModule

import copy
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdamWConfig:
    """See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html"""

    lr = 0.0006538379548447884

    betas = (0.9, 0.999)

    eps = 1e-8

    weight_decay = 1e-2


@dataclass
class PLModelConfig:
    only_train_heads: bool = True
    """ Train only attached head(s). """

    adamw: AdamWConfig = AdamWConfig()
    """ See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html """

    regularize_from_init: Optional[bool] = False
    """Regularize from init (base) model."""

    regularization_coef: Optional[float] = 0.1
    """Regularization from init coef."""


class LitBert(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        plc: PLModelConfig,
        data_module: BrainBiasDataModule,
    ):
        super().__init__()
        self._base_model = base_model
        self._plc = plc
        self._data_module = data_module
        if self._plc.regularize_from_init:
            self._init_params = copy.deepcopy([p for p in base_model.parameters()])
        self._init_heads()

    def _init_heads(self):
        self._heads = {}
        head_in_dim = self._base_model.config.hidden_size
        ds_cfg: DatasetConfig
        for ds_cfg, splits in self._data_module.ds_cfg_to_splits.items():
            ds: Union[IterableDataset, Dataset] = next(splits[s] for s in splits)
            label = ds.features[ds_cfg.label_col]
            if isinstance(label, ClassLabel):
                self._heads[ds_cfg] = nn.Linear(head_in_dim, label.num_classes)
            elif isinstance(label, Sequence):
                assert (
                    label.length > 0
                ), f"Expected positive length of label but {label.length=}"
                self._heads[ds_cfg] = nn.Linear(head_in_dim, label.length)
            elif isinstance(label, Value):
                self._heads[ds_cfg] = nn.Linear(head_in_dim, 1)
            else:
                raise NotImplemented()

            self.register_module(ds_cfg.name, self._heads[ds_cfg])

    def forward(self, tokens, mask):
        """Returns predictions per dataset per feature."""
        base_out = self._base_model(tokens, mask)
        base_out = base_out.last_hidden_state
        base_out = base_out[
            :, 0, :
        ]  # Only take the encoding of [CLS] -> [batch, d_model]
        return {ds_cfg: self._heads[ds_cfg](base_out) for ds_cfg in self._heads}

    def training_step(self, dl_batches, _):
        loss = 0
        for ds_cfg, batch in dl_batches.item():
            ds_cfg: DatasetConfig
            outputs = self.forward(batch["input_ids"], batch["attention_mask"])
            predictions: torch.Tensor = outputs[ds_cfg]
            targets = batch[ds_cfg.label_col]
            loss_fn = vars(F)[ds_cfg.loss_fn]
            loss += loss_fn(predictions, targets)

        if self._plc.regularize_from_init:
            reg_loss = 0
            params = self._base_model.parameters()
            for w, w0 in zip(params, self._init_params):
                w0 = w0.to(w.device)
                reg_loss += torch.pow(w - w0, 2).sum()
            loss += self._plc.regularization_coef * reg_loss

        # log and return
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        for ds_cfg, inner_b in batch.item():
            ds_cfg: DatasetConfig
            outputs = self.forward(inner_b["input_ids"], inner_b["attention_mask"])
            logits = outputs[ds_cfg]
            probs = F.softmax(logits, dim=-1)
            predicted_label = probs.argmax(dim=-1)
            targets = inner_b[ds_cfg.label_col]
            accuracy = (predicted_label == targets).float().mean()
            self.log("val_acc", accuracy)

    def test_step(self, batch, batch_idx):
        for ds_cfg, inner_b in batch.item():
            tokens = inner_b["input_ids"]
            masks = inner_b["attention_mask"]
            targets = inner_b[ds_cfg.label_col]
            output = self.forward(tokens, masks)
            logits = output[ds_cfg]
            probs = F.softmax(logits, dim=-1)
            predicted_label = probs.argmax(dim=-1)
            # log the accuracy (this automatically accumulates it over the whole test set)
            self.log(
                "test_acc",
                (predicted_label == targets).float().mean(),
                prog_bar=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        # ! FIXME this breaks if you first only train head and then train the whole thing
        if self._plc.only_train_heads:
            for param in self._base_model.parameters():
                param.requires_grad = False
        # TODO: Consider exclude Laynorm and other.
        # See BERT: https://github.com/google-research/bert/blob/master/optimization.py#L65
        optimizer = torch.optim.AdamW(self.parameters(), **vars(self._plc.adamw))
        return optimizer
