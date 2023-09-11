from dataclasses import dataclass
from typing import Optional, Union, Tuple
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

    lr: float = 1e-3
    """ Learn rate. Default for AdamW is 1e-3. """

    betas: Tuple[float, float] = (0.9, 0.999)

    eps: float = 1e-8

    weight_decay: float = 1e-2


@dataclass
class ReduceLROnPlateauConfig:
    patience: int = 10
    """ 
    Number of epochs with no improvement after which learning rate
    will be reduced. For example, if `patience = 2`, then we will
    ignore the first 2 epochs with no improvement, and will only decrease 
    the LR after the 3rd epoch if the loss still hasn't improved then.
    """

    factor: float = 0.1
    """
    Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """

    cooldown: int = 0
    """
    Number of epochs to wait before resuming normal operation after 
    lr has been reduced.
    """

    min_lr: float = 0
    """
    A scalar or a list of scalars. A lower bound on the learning rate
    of all param groups or each group respectively.
    """
    eps: float = 1e-8
    """
    Minimal decay applied to lr. If the difference between new and old lr
    is smaller than eps, the update is ignored. 
    """
    verbose: bool = True


@dataclass
class PLModelConfig:
    train_all: bool = False
    """ Train only attached head(s) or model and all heads. """

    adamw: AdamWConfig = AdamWConfig()
    """ See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html """

    has_ReduceLROnPlateau: bool = True

    reduceLROnPlateau_config: ReduceLROnPlateauConfig = ReduceLROnPlateauConfig()

    lr_scheduler_frequency: Optional[int] = 10
    """
    How many intervals (step/epoch) should pass. 1 corresponds to updating the 
    learning rate after every step/epoch.  It monitors "train_loss" metric.
    """

    lr_scheduler_interval : Optional[str] = 'epoch'

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
    VAL_ACC = "val_acc"

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
        self.log(PLModel.VAL_ACC, accuracy)

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
        if not self._plc.train_all:
            for param in self._base_model.parameters():
                param.requires_grad = False
        # TODO: Consider exclude Laynorm and other.
        # See BERT: https://github.com/google-research/bert/blob/master/optimization.py#L65
        optimizer = torch.optim.AdamW(self.parameters(), **vars(self._plc.adamw))
        if not self._plc.has_ReduceLROnPlateau:
            return optimizer
        lr_scheduler_config = {
            # See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **vars(self._plc.reduceLROnPlateau_config)
            ),
            "interval": self._plc.lr_scheduler_interval,
            "frequency": self._plc.lr_scheduler_frequency,
            "monitor": "train_loss",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
