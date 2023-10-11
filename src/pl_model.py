from collections import defaultdict
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
from torch.optim.lr_scheduler import StepLR, ConstantLR, SequentialLR

import copy
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    CosineSimilarity,
)
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score

STEPLR_STEPS_RATE = 0.02


@dataclass
class AdamWConfig:
    """See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html"""

    lr: float = 2e-5
    """ Learn rate. """

    betas: Tuple[float, float] = (0.9, 0.999)

    eps: float = 1e-8

    weight_decay: float = 1e-2


@dataclass
class PLModelConfig:
    train_all: bool = False
    """ Train only attached head(s) or model and all heads. """

    adamw: AdamWConfig = AdamWConfig()
    """ See https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html """

    has_learning_rate_decay: bool = True
    """ If to use LR linear decay. """

    lr_warm_up_steps: Optional[Union[int, float]] = 0.75
    """ 
    Steps before running linear LR decay. 
    Or a fraction of total steps for warming up, i.e. a float. 
    """

    regularize_from_init: Optional[bool] = False
    """Regularize from init (base) model."""

    regularization_coef: Optional[float] = 0.1
    """Regularization from init coef."""

    token_location: int = 0
    """ 
    Which token to use for prediction. Example: 0 - to take 
    [CLS] token for BERT like models.
    """

    stepLR_gamma: Optional[float] = 0.99
    """ Multiplicative factor of learning rate decay. """

    stepLR_step_size: Optional[int] = None
    f""" 
    Period of learning rate decay, in steps number. If not specified then it
    decays after each {STEPLR_STEPS_RATE * 100}% of estimated total steps.
    """

    batch_size_all: Optional[int] = None
    """ Batch size for all datasets broken evenly. """

    lr_base_model_factor : float = 1.0
    """ This factor multiplies the learning rate for the base model. """


class PLModel(pl.LightningModule):
    _VALIDATION = "validation"
    _TEST = "test"

    def __init__(
        self,
        plc: PLModelConfig,
        data_module: BrainBiasDataModule,
        base_model: nn.Module,
    ):
        super().__init__()
        self._base_model = base_model
        self._plc = plc
        self.data_module = data_module
        self.learning_rate = plc.adamw.lr
        if plc.batch_size_all is not None:
            self.data_module.batch_size = plc.batch_size_all
        if self._plc.regularize_from_init:
            self._init_params = copy.deepcopy([p for p in base_model.parameters()])
        self._init_heads()

    @property
    def batch_size(self):
        """Expose batch size for Pytorch Lightning LR finder."""
        return self.data_module.batch_size

    @batch_size.setter
    def batch_size(self, val):
        self.data_module.batch_size = val

    def _init_heads(self):
        self._heads = {}
        self._head_metrics = defaultdict(dict)
        self.main_val_metric = None

        head_in_dim = self._base_model.config.hidden_size
        ds_cfg: DatasetConfig
        # Warning. The order matters as the first one would be used for the early stopping.
        for ds_cfg, splits in self.data_module._cfg_to_datasets.items():
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
            self._init_metrics(ds_cfg, label)

    def _init_metrics(self, ds_cfg: DatasetConfig, label):
        # Warning. The order matters as the first one would be used for the early stopping.
        for split in (PLModel._VALIDATION, PLModel._TEST):
            collection: MetricCollection = None
            prefix = f"{ds_cfg.name}-{split}-"
            if isinstance(label, ClassLabel):
                collection = MetricCollection(
                    [
                        MulticlassAccuracy(num_classes=label.num_classes),
                        MulticlassAUROC(
                            num_classes=label.num_classes,
                            average="macro",
                            thresholds=None,
                        ),
                        MulticlassF1Score(num_classes=label.num_classes)
                    ],
                    prefix=prefix,
                )
            elif isinstance(label, Sequence):
                collection = MetricCollection(
                    [
                        MeanSquaredError(),
                        MeanAbsoluteError(),
                        CosineSimilarity(reduction="mean"),
                    ],
                    prefix=prefix,
                )

            if collection:
                name = f"{prefix}metrics"
                self.register_module(name, collection)
                self._head_metrics[ds_cfg][split] = (name, collection)

    def forward(self, tokens, mask):
        """Returns predictions per dataset per feature."""
        base_out = self._base_model(input_ids=tokens, attention_mask=mask)
        logits = base_out.last_hidden_state[:, self._plc.token_location, :]
        return {ds_cfg: self._heads[ds_cfg](logits) for ds_cfg in self._heads}

    def training_step(self, dl_batches, _):
        """
        Runs train loop. This expects multiple dataloaders, i.e. CombinedLoader
        with mode other than 'sequential'.
        """
        loss = 0
        total_batch_size = 0
        for ds_cfg, batch in dl_batches.items():
            ds_cfg: DatasetConfig
            if not batch:
                continue
            outputs = self.forward(batch["input_ids"], batch["attention_mask"])
            predictions: torch.Tensor = outputs[ds_cfg]
            targets = batch[ds_cfg.label_col]
            loss_fn = vars(F)[ds_cfg.loss_fn]
            loss += loss_fn(predictions, targets)
            total_batch_size += ds_cfg.train.batch_size

        if self._plc.regularize_from_init:
            reg_loss = 0
            params = self._base_model.parameters()
            for w, w0 in zip(params, self._init_params):
                w0 = w0.to(w.device)
                reg_loss += torch.pow(w - w0, 2).sum()
            loss += self._plc.regularization_coef * reg_loss

        # log and return
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=total_batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        Runs validation loop. This expects multiple dataloaders, i.e. CombinedLoader
        with the 'sequential' mode.
        """
        self._calc_metrics(PLModel._VALIDATION, batch, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        Runs test loop. This expects multiple dataloaders, i.e. CombinedLoader
        with the 'sequential' mode.
        """
        self._calc_metrics(PLModel._TEST, batch, dataloader_idx)

    def _calc_metrics(self, step_name, batch, dataloader_idx):
        if not batch:
            return
        ds_cfg = self.data_module.dataloader_idx_to_config[dataloader_idx]
        name, metric_col = self._head_metrics[ds_cfg][step_name]
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        predictions = outputs[ds_cfg]
        targets = batch[ds_cfg.label_col]
        m_result = metric_col(predictions, targets)
        self.log_dict(
            m_result,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self):
        self.trainer.estimated_stepping_batches
        if not self._plc.train_all:
            for param in self._base_model.parameters():
                param.requires_grad = False
        # TODO: Consider exclude Laynorm and other.
        # See BERT: https://github.com/google-research/bert/blob/master/optimization.py#L65
        optimizer = torch.optim.AdamW(
            [{"params": self._base_model.parameters(), "lr": self.learning_rate * self._plc.lr_base_model_factor}]
            + [{"params": h.parameters(), "lr": self.learning_rate} for h in self._heads.values()],
            lr=self.learning_rate,  # To facilitate LR finder.
            betas=self._plc.adamw.betas,
            eps=self._plc.adamw.eps,
            weight_decay=self._plc.adamw.weight_decay,
        )

        if self._plc.has_learning_rate_decay:
            if isinstance(self._plc.lr_warm_up_steps, int):
                warm_up_steps = self._plc.lr_warm_up_steps
            elif isinstance(self._plc.lr_warm_up_steps, float):
                warm_up_steps = int(
                    self.trainer.estimated_stepping_batches * self._plc.lr_warm_up_steps
                )
            else:
                warm_up_steps = self.trainer.estimated_stepping_batches

            constantlr = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=warm_up_steps,
            )
            step_size = self._plc.stepLR_step_size or int(
                self.trainer.estimated_stepping_batches * STEPLR_STEPS_RATE
            )
            llr = StepLR(
                optimizer,
                gamma=self._plc.stepLR_gamma,
                step_size=step_size,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": SequentialLR(
                        optimizer,
                        schedulers=[constantlr, llr],
                        milestones=[warm_up_steps],
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss",
                },
            }
        return optimizer

    @property
    def main_val_metric_name(self):
        # Take the first:
        ds_cfg: DatasetConfig = next(self.data_module._cfg_to_datasets)
        collection: MetricCollection
        _, collection = next(self._head_metrics[ds_cfg][PLModel._VALIDATION])
        mname, _ = next(iter(collection.items()))
        return mname
