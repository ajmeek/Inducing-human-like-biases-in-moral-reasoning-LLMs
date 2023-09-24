from .DS000212_LFB_Dataset import load_LFB_dataset
from .constants import Sampling
from datasets import IterableDataset, Dataset
from pathlib import Path
from typing import Union, List
import os

CONFIGURATIONS = ("learning_from_brains",)


def load(
    name: str = CONFIGURATIONS[0],
    sampling_method: str = Sampling.LAST,
    split: Union[List[str], str] = ["train"],
) -> List[Union[IterableDataset, Dataset]]:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    assert sampling_method in Sampling.ALL
    split = [split] if isinstance(split, str) else split
    split = [s for s in split if s]
    if any("[" in s or "]" in s for s in split):
        raise NotImplemented()
    assert len(split) == 1 and split[0] == "train", "Only 'train' split supported."

    if name == CONFIGURATIONS[0]:
        return [load_LFB_dataset(Path(base_dir), sampling_method=sampling_method)]

    raise NotImplemented(
        f'Not implemented for "{name}" configuration and {split} split.'
    )
