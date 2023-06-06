from csv import DictReader
from pathlib import Path
from pprint import pp
from torch.utils.data import IterableDataset
from typing import Dict, Tuple, Generator
import numpy as np
import torch
import webdataset as wds   
from re import search

class DS000212_LFB_Dataset(IterableDataset):
    """
    Map-style dataset that loads ds000212 dataset with its scenarios from disk and
    prepares it for fine tuning.
    """
    event_to_scenario = {
        "A_PHA": ("accidental", "Physical harm"),
        "B_PSA": ("accidental", "Psychological harm"),
        "C_IA": ("accidental", "Incest"),
        "D_PA": ("accidental", "Pathogen"),
        "E_NA": ("accidental", "Neutral"),
        "F_PHI": ("intentional", "Physical harm"),
        "G_PSI": ("intentional", "Psychological harm"),
        "H_II": ("intentional", "Incest"),
        "I_PI": ("intentional", "Pathogen"),
        "J_NI": ("intentional", "Neutral"),
    }

    def __init__(self, dataset_path: Path, scenarios_csv: Path, tokenizer):
        super().__init__()

        assert dataset_path.exists()
        assert scenarios_csv.exists()
        #assert tokenizer
        self.target_head_dim = None
        self._dataset_path = dataset_path
        self._init_scenarios(scenarios_csv)
        assert any(self._scenarios)
        self._tokenizer = tokenizer

        tarfiles=[str(f) for f in Path(dataset_path).glob('*.tar')]
        self.wdataset = wds.WebDataset(tarfiles).decode("pil").compose(self._get_samples)

    def __iter__(self):
        return iter(self.wdataset)

    def _init_scenarios(self, scenarios_csv: Path):
        self._scenarios = []
        with open(scenarios_csv, newline='', encoding='utf-8') as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self._scenarios.append(row)

    def _parse_label(self, label) -> str:
        condition, item, key = label
        if condition in DS000212_LFB_Dataset.event_to_scenario:
            skind, stype = DS000212_LFB_Dataset.event_to_scenario[condition]
            found = [s for s in self._scenarios if s['item'] == item]
            if not found:
                return None
            found = found[0]
            assert found['type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {event}."
            text = ' '.join([
                found['background'],
                found['action'],
                found['outcome'],
                found[skind]
            ])
            return text

    def _get_samples(self, src):
        for sample in src:
            #key = sample['__key__']
            out = dict()
            bold = None
            for key, value in sample.items():
                #print(f" {key=} {value=}")
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
                    #out['__key__'] = f"{key} {start}-{end}"
                    out['start'] = start
                    out['end'] = end
                    out["inputs"] = torch.from_numpy(bold[start:end+1]).to(torch.float)
                    out['label'] = self._parse_label(label)
                    yield out.copy()

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



if __name__ == '__main__':
    from pprint import pp
    ds = DS000212_LFB_Dataset(Path('./data/ds000212_learning-from-brains/'), Path('./data/ds000212_scenarios.csv'), None)
    count = 50
    for item in ds:
        pp(item)
        count -= 1
        if count < 1:
            break