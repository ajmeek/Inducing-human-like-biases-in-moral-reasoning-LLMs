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
    train_full: bool = False
    """ Train only attached head(s). """

    adamw: AdamWConfig = AdamWConfig()
    """ See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html """

    regularize_from_init: Optional[bool] = False
    """Regularize from init (base) model."""

    regularization_coef: Optional[float] = 0.1
    """Regularization from init coef."""

    token_location: int = 0
    """ 
    Which token to use for prediction. Example: 0 - to take 
    [CLS] token for BERT like models.
    """


class PLModel(pl.LightningModule):
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
                head = nn.Linear(head_in_dim, label.num_classes)
            elif isinstance(label, Sequence):
                assert (
                    label.length > 0
                ), f"Expected positive length of label but {label.length=}"
                head = nn.Linear(head_in_dim, label.length)
            elif isinstance(label, Value):
                head = nn.Linear(head_in_dim, 1)
            else:
                raise NotImplemented()

            self._heads[ds_cfg] = head
            self.register_module(ds_cfg.name, head)

    def forward(self, tokens, mask):
        """Returns predictions per dataset per feature."""
        base_out = self._base_model(tokens, mask)
        base_out = base_out.last_hidden_state
        base_out = base_out[:, self._plc.token_location, :]
        return {ds_cfg: self._heads[ds_cfg](base_out) for ds_cfg in self._heads}

    def training_step(self, dl_batches, _):
        """
        Runs train loop. This expects multiple dataloaders, i.e. CombinedLoader
        with mode other than 'sequential'.
        """
        loss = 0
        for ds_cfg, batch in dl_batches.items():
            if not batch:
                continue
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

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        Runs validation loop. This expects multiple dataloaders, i.e. CombinedLoader
        with the 'sequential' mode.
        """
        if not batch:
            return
        ds_cfg: DatasetConfig = self._data_module.dataloader_idx_to_config[
            dataloader_idx
        ]
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        logits = outputs[ds_cfg]
        probs = F.softmax(logits, dim=-1)
        predicted_label = probs.argmax(dim=-1)
        targets = batch[ds_cfg.label_col]
        accuracy = (predicted_label == targets).float().mean()
        self.log("val_acc", accuracy)

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        Runs test loop. This expects multiple dataloaders, i.e. CombinedLoader
        with the 'sequential' mode.
        """
        if not batch:
            return
        ds_cfg: DatasetConfig = self._data_module.dataloader_idx_to_config[
            dataloader_idx
        ]
        tokens = batch["input_ids"]
        masks = batch["attention_mask"]
        targets = batch[ds_cfg.label_col]
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
        if not self._plc.train_full:
            for param in self._base_model.parameters():
                param.requires_grad = False
        # TODO: Consider exclude Laynorm and other.
        # See BERT: https://github.com/google-research/bert/blob/master/optimization.py#L65
        optimizer = torch.optim.AdamW(self.parameters(), **vars(self._plc.adamw))
        return optimizer
