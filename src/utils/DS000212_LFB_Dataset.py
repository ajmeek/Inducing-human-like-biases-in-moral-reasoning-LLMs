from csv import DictReader
from pathlib import Path
from pprint import pp
from torch.utils.data import IterableDataset
from typing import Dict, Tuple, Generator
import numpy as np
import torch
from utils.DS000212Scenarios import DS000212Scenarios
import webdataset as wds   
from re import search

class DS000212_LFB_Dataset(IterableDataset):
    """
    Map-style dataset that loads ds000212 dataset with its scenarios from disk and
    prepares it for fine tuning.
    """
    def __init__(self, dataset_path: Path, scenarios_csv: Path, tokenizer, subject=None,
                 intervals=(-1,)):
        super().__init__()

        assert dataset_path.exists()
        assert scenarios_csv.exists()
        self.target_head_dim = None
        self._dataset_path = dataset_path
        self._tokenizer = tokenizer
        self._intervals = intervals

        if subject is not None:
            tarfiles=[str(f) for f in Path(dataset_path).glob(f'*{subject}*.tar')]
        else:
            tarfiles=[str(f) for f in Path(dataset_path).glob('*.tar')]
        self.wdataset = wds.WebDataset(tarfiles).decode("pil").compose(self._get_samples)

        self.target_head_dim = 1024
        self._scenarios = DS000212Scenarios(scenarios_csv)

    def __iter__(self):
        return iter(self.wdataset)

    def _get_samples(self, src):
        for sample in src:
            out = dict()
            bold = None
            for key, value in sample.items():
                if key == "bold.pyd":
                    bold = np.array(value).astype(float)
                else:
                    out[key] = value

            if bold is not None:
                key = out['__key__']
                tsvfile = Path(self._dataset_path / f"{key}.tsv")
                if not tsvfile.exists():
                    continue
                data_items, labels = self._process_tsv(tsvfile)
                for (start, end), label in zip(data_items, labels):
                    text = self._scenarios.parse_label(label, len_intervals=len(self._intervals))
                    if not text:
                        continue
                    tokens = None
                    mask = None
                    if self._tokenizer is not None:
                        tokenized = self._tokenizer(text, padding='max_length', truncation=True)
                        tokens = torch.tensor(tokenized['input_ids'])
                        mask = torch.tensor(tokenized['attention_mask'])
                    target = self._sample(bold[start:end])
                    if len(self._intervals) == 1:
                        assert target.shape == (self.target_head_dim,), f"target.shape: {target.shape}"
                    else:
                        assert target.shape == (len(self._intervals), self.target_head_dim), f"target.shape: {target.shape}"
                    yield tokens, mask, target

    def _sample(self, bold_sequence : np.array) -> torch.Tensor:
        # TR = 2
        # react_time = 3 // TR
        # intervals = [2, 4, 6, 8]
        return torch.from_numpy(bold_sequence[self._intervals]).to(torch.float)
        # torch.from_numpy(bold_sequence[-react_time]).to(torch.float)
    
    def _process_tsv(self, from_tsv: Path):
        scenarios = []
        with open(from_tsv, newline='') as csvfile:
            scenarios = list(DictReader(csvfile, delimiter='\t', quotechar='"'))
        TR = 2
        data_items = []
        labels = []
        for s in scenarios:
            onset = s['onset']
            try:
                if '[' in onset and ']' in onset:
                    m = search('\d+', onset)
                    assert m
                    onset = int(m.group(0))
                else:
                    onset = int(onset)
            except Exception as e:
                print(f'Failed to parse "onset" for {from_tsv}: {e}')
                continue
            duration = int(s['duration'])
            onset //= TR
            duration //= TR
            hemodynamic_lag = 6 // TR
            data_items.append((onset+hemodynamic_lag, onset+duration+hemodynamic_lag))
            label = [s[k] for k in ('condition', 'item', 'key')]
            labels.append(label)
        data_items = np.array(data_items)
        labels = np.array(labels)
        return data_items, labels
