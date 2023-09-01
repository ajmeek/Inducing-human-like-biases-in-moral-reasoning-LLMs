from datasets import Dataset, IterableDataset, ClassLabel, Value, ClassLabel, Sequence
from utils.BrainBiasDataModule import DatasetConfig, BrainBiasDataModule
import torch.nn as nn
from typing import Union


class BERT(nn.Module):
    def __init__(self, base_model, data_module: BrainBiasDataModule):
        super().__init__()
        self.base = base_model
        head_in_dim = base_model.config.hidden_size

        self._heads = {}
        ds_cfg: DatasetConfig
        for ds_cfg, splits in data_module.ds_cfg_to_splits.items():
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
        base_out = self.base(tokens, mask)
        base_out = base_out.last_hidden_state
        base_out = base_out[
            :, 0, :
        ]  # Only take the encoding of [CLS] -> [batch, d_model]
        return {ds_cfg: self._heads[ds_cfg](base_out) for ds_cfg in self._heads}
