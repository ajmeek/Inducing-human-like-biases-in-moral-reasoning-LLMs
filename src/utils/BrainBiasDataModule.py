from dataclasses import dataclass
from datasets import load_dataset, Dataset, IterableDataset, Split
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from typing import Optional, Union, List, Tuple
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch import LightningDataModule


@dataclass(frozen=False)
class SplitConfig:
    batch_size: Optional[int] = 1
    """ How many samples per batch to load.  """

    shuffle: Optional[bool] = False
    """ Set to ``True`` to have the data reshuffled at every epoch.  """

    slicing: str = None
    """ 
    Split slicing specification. See https://www.tensorflow.org/datasets/splits . 
    """


@dataclass(frozen=False)
class DatasetConfig:
    train: Optional[SplitConfig]
    """ Relates to Split.TRAIN """

    validation: Optional[SplitConfig]
    """ Relates to Split.VALIDATION """

    test: Optional[SplitConfig]
    """ Relates to Split.TEST """

    enable: Optional[bool] = True
    """ Whether to use this dataset."""

    path: str = None
    """ Path to a local folder or remote url. """

    name: str = None
    """ Configuration of dataset. """

    label_col: str = "label"
    """ Name of the target column. """

    input_col: str = "input"
    """ Name of the input column. """

    loss_fn: str = "cross_entropy"
    """ Loss function as given in torch.nn.functional namespace. """

    revision: str = None
    """
    Version of the dataset script to load.
    As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
    You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
    """

    def get_split_spec(self, split: str) -> str:
        cd = vars(self)  # Dataset config dictionary.
        if split not in cd or not cd[split]:
            return None
        si: SplitConfig = cd[split]
        if not si.slicing:
            return split
        assert (
            "[" in si.slicing and "]" in si.slicing
        ), f'Invalid slicing format: "{si.slicing}"'
        return f"{split}{si.slicing}"

    def __hash__(self) -> int:
        return hash((self.path or "") + (self.name or ""))


class BrainBiasDataModule(LightningDataModule):
    TRAIN_VAL_TEST = (Split.TRAIN, Split.VALIDATION, Split.TEST)

    def __init__(
        self, ds_configs: List[DatasetConfig], tokenizer, num_workers: int = 0
    ) -> None:
        super().__init__()
        self._num_workers = num_workers
        self._ds_configs = ds_configs
        self.tokenizer = tokenizer
        self.dataloader_idx_to_config = []
        self._load_datasets()
        assert self.batch_size > 0

    @property
    def batch_size(self):
        return sum(
            c.train.batch_size
            for c in self._ds_configs
            if c.train and c.train.batch_size
        )

    @batch_size.setter
    def batch_size(self, val):
        """
        Overrides batch size in all DatasetConfigs for all splits.
        The given value is broken evenly with the remaining left for the last.
        """

        assert val > 0
        for split in BrainBiasDataModule.TRAIN_VAL_TEST:
            split = str(split)
            n = sum(
                1
                for c in self._ds_configs
                if vars(c)[split] and vars(c)[split].batch_size > 0
            )
            assert (
                val >= n
            ), "Expect batch size to be greater than or equal to the number of datasets."
            m = val // n
            r = val % n
            assert m > 0
            for i, cfg in enumerate(self._ds_configs):
                s_cfg = vars(cfg)[split]
                if not s_cfg:
                    continue
                s_cfg.batch_size = m + (r if i + 1 == n else 0)
            assert (
                sum(
                    vars(c)[split].batch_size
                    for c in self._ds_configs
                    if vars(c)[split] and vars(c)[split].batch_size > 0
                )
                == val
            )

    def _load_datasets(self):
        """Load datasets splits into memory."""

        self._cfg_to_datasets = {}
        for cfg in self._ds_configs:
            # Load dataset splits and prepare for the following methods:
            self._cfg_to_datasets[cfg] = {}
            split_spec = {
                split: s_spec
                for split in BrainBiasDataModule.TRAIN_VAL_TEST
                for s_spec in (cfg.get_split_spec(str(split)),)
                if s_spec
            }
            ds_array: List[Union[IterableDataset, Dataset]] = load_dataset(
                path=cfg.path,
                name=cfg.name,
                split=list(split_spec.values()),
                revision=cfg.revision,
            )
            for idx, split in enumerate(split_spec.keys()):
                self._cfg_to_datasets[cfg][str(split)] = ds_array[idx]

    def setup(self, stage):
        """Preprocess datasets splits before creating DataLoaders."""

        def _create_map(cfg):
            def map_(batch):
                # TODO make sure it doesn't add SEP tokens when there's a full stop
                d = self.tokenizer(
                    batch[cfg.input_col], padding="max_length", truncation=False
                )
                return d

            return map_

        def _filter(batch):
            return [
                len(e) == self.tokenizer.model_max_length for e in batch["input_ids"]
            ]

        for cfg, splits in self._cfg_to_datasets.items():
            for sname in splits:
                ds: Union[IterableDataset, Dataset] = splits[sname]
                splits[sname] = (
                    ds.map(_create_map(cfg), batched=True, batch_size=10000)
                    .filter(_filter, batched=True, batch_size=10000)
                    .with_format("torch")
                )

    def _create_dataloaders(self, split: Split):
        s = str(split)
        # TODO: Try shuffle at Dataset level not Dataloader
        # See https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable#speed-differences
        res = {
            cfg: DataLoader(
                dss[s],
                batch_size=s_cfg.batch_size,
                # This avoids "num_samples should be a positive integer" error:
                shuffle=(s_cfg.shuffle if len(dss[s]) > 0 else False),
                num_workers=self._num_workers,
            )
            for cfg, dss in self._cfg_to_datasets.items()
            for s_cfg in (vars(cfg)[s],)
            # Warning. If len(dss[s])==0 still add this as dataloader_idx might be
            # equal to the one from other stage (from train in validation), so that
            # wrong DatasetConfig taken.
            if s in dss and dss[s] is not None
        }
        self.dataloader_idx_to_config = list(res.keys())
        return res

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # Warning. Currently Lightning doesn't support any mode for
        # CombinedLoader in all methods (train, validataion, test).
        # See https://lightning.ai/docs/pytorch/stable/data/iterables.html#multiple-iterables
        return CombinedLoader(self._create_dataloaders(Split.TRAIN), mode="max_size")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return CombinedLoader(
            self._create_dataloaders(Split.VALIDATION), mode="sequential"
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return CombinedLoader(self._create_dataloaders(Split.TEST), mode="sequential")
