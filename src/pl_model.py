from dataclasses import dataclass
from typing import Optional
from utils.BrainBiasDataModule import DatasetConfig
from datasets import Split
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


# lightning wrapper for training (scaling, parallelization etc)
class LitBert(pl.LightningModule):
    def __init__(
        self, model: nn.Module, plc: PLModelConfig, data_module: BrainBiasDataModule
    ):
        super().__init__()
        self._model: nn.Module = model
        self._plc = plc
        self._data_module = data_module
        if self._plc.regularize_from_init:
            self.init_params = copy.deepcopy([p for p in model.base.parameters()])

    def training_step(self, dl_batches, _):
        loss = 0
        for dl_idx, batch in enumerate(dl_batches):
            ds_cfg: DatasetConfig = self._get_ds_cfg_by(Split.TRAIN, dl_idx)
            outputs = self._model(batch["input_ids"], batch["attention_mask"])
            predictions: torch.Tensor = outputs[ds_cfg]
            targets = batch[ds_cfg.label_col]
            loss_fn = vars(F)[ds_cfg.loss_fn]
            loss += loss_fn(predictions, targets)

        if self._plc.regularize_from_init:
            reg_loss = 0
            params = self._model.base.parameters()
            for w, w0 in zip(params, self.init_params):
                w0 = w0.to(w.device)
                reg_loss += torch.pow(w - w0, 2).sum()
            loss += self._plc.regularization_coef * reg_loss

        # log and return
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, dataloader_idx: int = 0):
        ds_cfg: DatasetConfig = self._get_ds_cfg_by(Split.VALIDATION, dataloader_idx)
        outputs = self._model(batch["input_ids"], batch["attention_mask"])
        logits = outputs[ds_cfg]
        probs = F.softmax(logits, dim=-1)
        predicted_label = probs.argmax(dim=-1)
        targets = batch[ds_cfg.label_col]
        accuracy = (predicted_label == targets).float().mean()
        self.log("val_acc", accuracy)

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        ds_cfg: DatasetConfig = self._get_ds_cfg_by(Split.TEST, dataloader_idx)
        tokens = batch["input_ids"]  # Tokens in a batch.
        masks = batch["attention_mask"]
        targets = batch[ds_cfg.label_col]
        output = self._model(tokens, masks)
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
        return predicted_label

    def configure_optimizers(self):
        # ! FIXME this breaks if you first only train head and then train the whole thing
        if self._plc.only_train_heads:
            for param in self._model.base.parameters():
                param.requires_grad = False
        # TODO: Consider exclude Laynorm and other.
        # See BERT: https://github.com/google-research/bert/blob/master/optimization.py#L65
        optimizer = torch.optim.AdamW(self.parameters(), **vars(self._plc.adamw))
        return optimizer

    def _get_ds_cfg_by(self, split: str, dl_idx: int) -> DatasetConfig:
        if not split in self._data_module.dataloader_idx_to_ds_cfg:
            return None
        s_to_ds = self._data_module.dataloader_idx_to_ds_cfg[split]
        if dl_idx >= len(s_to_ds):
            return None
        return s_to_ds[dl_idx]
