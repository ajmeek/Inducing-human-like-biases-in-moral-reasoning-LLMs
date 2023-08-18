from csv import DictReader
from pathlib import Path
from pprint import pp
from torch.utils.data import IterableDataset
from typing import Dict, Tuple, Generator
import numpy as np
import torch
from utils.constants import Sampling, FMRI, DS000212
from utils.DS000212Scenarios import DS000212Scenarios
#from DS000212Scenarios import DS000212Scenarios
import webdataset as wds   
from re import search

class DS000212_LFB_Dataset(IterableDataset):
    """
    Map-style dataset that loads ds000212 dataset, Learning From Brains (LFB), with
    its scenarios from disk and prepares it for fine tuning.
    """

    @staticmethod
    def sample_from_bold_sequence(sequence : np.array, method : Sampling) -> torch.Tensor:
        if method in Sampling.LAST:
            result = sequence[-FMRI.REACT_TIME]
        elif method is Sampling.AVG:
            result = np.average(sequence, axis=-2)
        elif method is Sampling.MIDDLE:
            result = sequence[len(sequence) // 2]
        elif method is Sampling.SENTENCES:
            result = sequence[DS000212.Periods.ENDS[:3] + [-FMRI.REACT_TIME]]
        else:
            raise NotImplementedError()
        return torch.from_numpy(result).to(torch.float)

    def __init__(self, context, tokenizer, subject=None):
        datapath = context['datapath']
        dataset_path = datapath / 'ds000212_learning-from-brains'
        scenarios_csv = datapath / 'ds000212_scenarios.csv'

        assert dataset_path.exists(), f"No dataset found at '{dataset_path} (hint: run.sh datasets)"
        assert scenarios_csv.exists()
        self.head_dims = None
        self._dataset_path = dataset_path
        self._tokenizer = tokenizer
        self._context = context

        if subject is not None:
            tarfiles=[str(f) for f in Path(dataset_path).glob(f'*{subject}*.tar')]
        else:
            tarfiles=[str(f) for f in Path(dataset_path).glob('*.tar')]
        self.wdataset = wds.WebDataset(tarfiles, nodesplitter=wds.shardlists.split_by_worker).decode("pil").compose(self._get_samples)

        self.head_dims = 1024
        self._scenarios = DS000212Scenarios(scenarios_csv, context)
        super().__init__()

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
                    text = self._scenarios.parse_label(label)
                    if not text:
                        continue
                    tokens = None
                    mask = None
                    if self._tokenizer is not None:
                        tokenized = self._tokenizer(text, padding='max_length', truncation=True)
                        tokens = torch.tensor(tokenized['input_ids'])
                        mask = torch.tensor(tokenized['attention_mask'])
                    target = DS000212_LFB_Dataset.sample_from_bold_sequence(bold[start:end], self._context['sampling_method'])
                    error_msg = f'expect each sample has its label but ({target.shape=}, {tokens.shape=})'
                    if target.ndim == 1:
                        assert target.ndim == tokens.ndim, error_msg
                        yield tokens, mask, target
                    else:
                        assert target.size(0) == tokens.size(0), error_msg
                        for i in range(target.size(0)):
                            yield tokens[i], mask[i], target[i]



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
