from pl_model import PLModelConfig
from utils.BrainBiasDataModule import (
    DatasetConfig,
    FMRIDatasetConfig,
    HFDatasetConfig,
    SplitConfig,
)


from simple_parsing import Serializable, field


from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class PLTrainerConfig:
    max_epochs: Optional[int] = 1
    """
    Stop training once this number of epochs is reached.  If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.  To enable infinite training, set ``max_epochs = -1``.
    """

    min_epochs: Optional[int] = None
    """ Force training for at least these many epochs. Disabled by default (None)."""

    max_steps: int = -1
    """
    Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
        and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set ``max_epochs`` to ``-1``.
    """

    min_steps: Optional[int] = None
    """ Force training for at least these number of steps. Disabled by default (``None``)."""

    max_time: Optional[str] = None
    """ 
    Stop training after this amount of time has passed. Disabled by default (``None``).
        The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds) 
    """

    limit_train_batches: Optional[Union[int, float]] = 15
    """ How much of training dataset to check (float = fraction, int = num_batches).
    """

    limit_val_batches: Optional[Union[int, float]] = 1.0
    """
    How much of validation dataset to check (float = fraction, int = num_batches).
    """

    limit_test_batches: Optional[Union[int, float]] = 1.0
    """
    How much of test dataset to check (float = fraction, int = num_batches).
    """

    val_check_interval: Optional[Union[int, float]] = 1.0
    """
    How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] 
    to check after a fraction of the training epoch. Pass an ``int`` to check after 
    a fixed number of training batches. An ``int`` value can only be higher than the 
    number of training batches when ``check_val_every_n_epoch=None``, which validates 
    after every ``N`` training batches across epochs or during iteration-based training.
        Default: ``1.0``.
    """

    check_val_every_n_epoch: Optional[int] = 1
    """
    Perform a validation loop every after every `N` training epochs. If ``None``,
    validation will be done solely based on the number of training batches, 
    requiring ``val_check_interval`` to be an integer value.
        Default: ``1``.
    """

    num_sanity_val_steps: Optional[int] = None
    """
    Sanity check runs n validation batches before starting the training routine.
        Set it to `-1` to run all batches in all validation dataloaders.
        Default: ``2``.
    """

    log_every_n_steps: Optional[int] = None
    """How often to log within steps.  Default: ``50``.  """

    enable_checkpointing: Optional[bool] = False
    """ If ``True``, enable checkpointing.
        It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`.
    """

    overfit_batches: Union[int, float] = 0.0
    """ Overfit a fraction of training/validation data (float) or a set number of batches (int). """

    precision: str = "32-true"
    """
    Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        Can be used on CPU, GPU, TPUs, HPUs or IPUs.
    """

    limit_train_batches: Optional[Union[int, float]] = 1.0
    """
    How much of training dataset to check (float = fraction, int = num_batches).
    """

    limit_val_batches: Optional[Union[int, float]] = 1.0
    """
    How much of validation dataset to check (float = fraction, int = num_batches).
    """

    limit_test_batches: Optional[Union[int, float]] = 1.0
    """
        How much of test dataset to check (float = fraction, int = num_batches).
    """


@dataclass
class Context(Serializable):
    model_path: str = "bert-base-cased"
    """ Huggingface model path or name. 
        See https://huggingface.co/docs/transformers/v4.32.1/en/model_doc/auto#transformers.AutoModel.from_pretrained
     """

    ds1: HFDatasetConfig = HFDatasetConfig(
        path="hendrycks/ethics", #should be ../.. from hpc script directory
        name="commonsense",
        revision="refs/pr/3",
        train=SplitConfig(batch_size=50, shuffle=True, slicing="[:100]"),
        validation=SplitConfig(batch_size=50, shuffle=False, slicing="[:100]"),
        test=SplitConfig(batch_size=50, shuffle=False, slicing="[:100]"),
        loss_fn="cross_entropy",
    )

    ds2: FMRIDatasetConfig = field(
        default_factory=lambda: FMRIDatasetConfig(
            path="../../data/ds000212",
            name="learning_from_brains",
            sampling_method="LAST",
            train=SplitConfig(batch_size=2, shuffle=False),
            validation=None,
            test=None,
            loss_fn="mse_loss",
        )
    )

    pltc: PLTrainerConfig = PLTrainerConfig()
    """ Lightning Trainer configuration.  """

    plc: PLModelConfig = PLModelConfig()
    """ PyTorch Lightning model configuration. """

    artifactspath: Path = Path(environ.get("AISCBB_ARTIFACTS_DIR", "./artifacts"))
    """ Path to the folder for artifacts. """

    datapath: Path = Path(environ.get("AISCBB_DATA_DIR", "./data"))
    """ Path to the folder with datasets.  """

    to_save_model: bool = False
    """ Whether to save checkpoint of the model after the test. """

    early_stop_threshold: Optional[float] = None
    """ 
    Monitor validation accuracy and stop when it reaches the
    threshold. If not set then no early stopping.
    """

    def get_ds_configs(self) -> List[DatasetConfig]:
        return [ds for ds in (self.ds1, self.ds2) if ds.enable]
