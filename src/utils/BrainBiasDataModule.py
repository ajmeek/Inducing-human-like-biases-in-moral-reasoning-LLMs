from dataclasses import dataclass
from datasets import load_dataset, Dataset, IterableDataset, Split
from functools import cache
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, Union, List
import importlib
import lightning.pytorch as pl
import sys


@dataclass(frozen=True)
class SplitConfig:
    batch_size: Optional[int] = 1
    """ How many samples per batch to load.  """

    shuffle: Optional[bool] = False
    """ Set to ``True`` to have the data reshuffled at every epoch.  """

    slicing: str = None
    """ 
    Split slicing specification. See https://www.tensorflow.org/datasets/splits . 
    In square parantheses. Example: '[:50%]' - half of a split, '[50:-100]' - from 50th sample till 100th. 

    """


@dataclass(frozen=True)
class DatasetConfig:
    train: Optional[SplitConfig]
    """ Relates to Split.TRAIN """

    validation: Optional[SplitConfig]
    """ Relates to Split.VALIDATION """

    test: Optional[SplitConfig]
    """ Relates to Split.TEST """

    path: str = None
    """ Path to a local folder or remote url. """

    name: str = None
    """ Configuration of dataset. """

    label_col: str = "label"

    loss_fn: str = "cross_entropy"
    """ Loss function as given in torch.nn.functional namespace. """

    def __hash__(self) -> int:
        return hash((self.path or "") + (self.name or ""))


@dataclass(frozen=True)
class FMRIDatasetConfig(DatasetConfig):
    """Load custom dataset from a local folder."""

    sampling_method: str = "LAST"
    """ Sampling method. """


@dataclass(frozen=True)
class HFDatasetConfig(DatasetConfig):
    """Load a dataset from the Hugging Face Hub, or a local dataset."""

    revision: str = None
    """
    Version of the dataset script to load.
    As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
    You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
    """


class BrainBiasDataModule(pl.LightningDataModule):
    def __init__(self, ds_configs: List[DatasetConfig], tokenizer) -> None:
        super().__init__()
        self._ds_configs = ds_configs
        self._tokenizer = tokenizer
        self.dataloader_idx_to_ds_cfg = {}
        self._load_datasets()

    def _load_datasets(self):
        """Load datasets splits into memory."""

        def _get_split_spec(c, s):
            s = str(s)
            cd = vars(c)
            if s not in cd or not cd[s]:
                return None
            si: SplitConfig = cd[s]
            if not si.slicing:
                return s
            assert (
                "[" in si.slicing and "]" in si.slicing
            ), f'Invalid slicing format: "{si.slicing}"'
            return f"{s}{si.slicing}"

        self.ds_cfg_to_splits = {}
        train_validation_test = (Split.TRAIN, Split.VALIDATION, Split.TEST)
        for ds_config in self._ds_configs:
            self.ds_cfg_to_splits[ds_config] = {}
            split_spec = [
                _get_split_spec(ds_config, split) for split in train_validation_test
            ]
            if isinstance(ds_config, FMRIDatasetConfig):
                ds_config: FMRIDatasetConfig
                module = self._import_module(ds_config)
                ds_array: List[Union[IterableDataset, Dataset]] = module.load(
                    name=ds_config.name,
                    sampling_method=ds_config.sampling_method,
                    split=split_spec,
                )
            elif isinstance(ds_config, HFDatasetConfig):
                ds_config: HFDatasetConfig
                ds_array: List[Union[IterableDataset, Dataset]] = load_dataset(
                    path=ds_config.path,
                    name=ds_config.name,
                    revision=ds_config.revision,
                    split=split_spec,
                )
            else:
                raise NotImplemented()

            for idx, split in enumerate(train_validation_test):
                if idx < len(ds_array):
                    self.ds_cfg_to_splits[ds_config][str(split)] = ds_array[idx]

    @cache
    def _import_module(self, dsconfig: FMRIDatasetConfig):
        dsconfig: FMRIDatasetConfig
        path = Path(dsconfig.path)
        assert path.exists() and path.is_dir()
        path = path.resolve()
        if not path in sys.path:
            sys.path.append(str(path))
        module_name = path.parts[-1]
        return importlib.import_module(module_name)

    def setup(self, stage):
        """Preprocess datasets splits before creating DataLoaders."""

        def _map(batch):
            # TODO make sure it doesn't add SEP tokens when there's a full stop
            d = self._tokenizer(batch["input"], padding="max_length", truncation=False)
            return d

        def _filter(batch):
            return [
                len(e) == self._tokenizer.model_max_length for e in batch["input_ids"]
            ]

        for _, splits in self.ds_cfg_to_splits.items():
            for s in splits:
                if not splits[s]:
                    continue
                ds: Union[IterableDataset, Dataset] = splits[s]
                splits[s] = (
                    ds.map(_map, batched=True, batch_size=10000)
                    .filter(_filter, batched=True, batch_size=10000)
                    .with_format("torch")
                )

    def _dataloaders(self, split: Split):
        """Create DataLoaders for all dataset per the given split."""
        s = str(split)
        res = []
        for ds_cfg, splits in self.ds_cfg_to_splits.items():
            if s not in splits or not splits[s]:
                continue
            s_cfg = vars(ds_cfg)[s]
            res.append(
                (
                    ds_cfg,
                    DataLoader(
                        splits[s], batch_size=s_cfg.batch_size, shuffle=s_cfg.shuffle
                    ),
                )
            )
        self.dataloader_idx_to_ds_cfg[s] = [cfg for cfg, _ in res]
        return [dl for _, dl in res]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return list(self._dataloaders(Split.TRAIN))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return list(self._dataloaders(Split.VALIDATION))

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return list(self._dataloaders(Split.TEST))
