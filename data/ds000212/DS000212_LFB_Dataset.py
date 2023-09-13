from .DS000212Scenarios import DS000212Scenarios
from .constants import Sampling, FMRI, DS000212
from csv import DictReader
from datasets import IterableDataset, Features, Value, DatasetInfo, Sequence, ClassLabel
from pathlib import Path
from re import search
import numpy as np
import webdataset as wds

DATA_LENGTH = 1024


def load_LFB_dataset(basedir: Path, sampling_method=Sampling.LAST, subject=None):
    """
    Map-style dataset that loads ds000212 dataset, Learning From Brains (LFB), with
    its scenarios from disk and prepares it for fine tuning.
    """

    def _sample_from_bold_sequence(sequence: np.array, method: Sampling) -> np.array:
        if method == Sampling.LAST:
            result = sequence[-FMRI.REACT_TIME]
        elif method == Sampling.AVG:
            result = np.average(sequence, axis=-2)
        elif method == Sampling.MIDDLE:
            result = sequence[len(sequence) // 2]
        elif method == Sampling.SENTENCES:
            result = sequence[DS000212.Periods.ENDS[:3] + [-FMRI.REACT_TIME]]
        else:
            raise NotImplementedError()
        return result

    def _get_samples(src):
        def _to_behavior_key(key):
            return int(key) - 1

        nonlocal scenarios
        for sample in src:
            out = dict()
            bold = None
            for key, value in sample.items():
                if key == "bold.pyd":
                    bold = np.array(value).astype(float)
                else:
                    out[key] = value

            if bold is not None:
                key = out["__key__"]
                tsvfile = Path(dataset_path / f"{key}.tsv")
                if not tsvfile.exists():
                    continue
                data_items, labels = _process_tsv(tsvfile)
                last_end = data_items[-1][-1]
                if bold.shape[0] <= last_end:
                    # Skip this run. Assume broken data.
                    continue
                for (start, end), label in zip(data_items, labels):
                    condition, item, behavior_key = label
                    text = scenarios.parse_label(condition, item)
                    if not text:
                        continue
                    target: np.array = _sample_from_bold_sequence(
                        bold[start:end], sampling_method
                    )
                    error_msg = f"expect each sample has its label but ({len(text)=}, {target.shape=})"
                    if target.ndim == 1:
                        assert (
                            len(target) == DATA_LENGTH
                        ), f"{len(target)=} == {DATA_LENGTH=}"
                        assert target.ndim == 1 and isinstance(text[0], str), error_msg
                        yield {
                            "label": target,
                            "input": text,
                            "behavior": _to_behavior_key(behavior_key),
                        }
                    else:
                        assert target.shape[0] == len(text), error_msg
                        for i in range(target.shape[0]):
                            assert (
                                len(target[i]) == DATA_LENGTH
                            ), f"{len(target[i])=} == {DATA_LENGTH=}"
                            yield {
                                "label": target[i],
                                "input": text[i],
                                "behavior": _to_behavior_key(behavior_key),
                            }

    def _process_tsv(from_tsv: Path):
        rows = []
        with open(from_tsv, newline="") as csvfile:
            rows = list(DictReader(csvfile, delimiter="\t", quotechar='"'))
        data_items = []
        labels = []
        for row in rows:
            onset = row["onset"]
            try:
                if "[" in onset and "]" in onset:
                    m = search("\d+", onset)
                    assert m
                    onset = int(m.group(0))
                else:
                    onset = int(onset)
            except Exception as e:
                print(f'Failed to parse "onset" for {from_tsv}: {e}')
                continue
            duration = int(row["duration"])
            onset //= FMRI.TR
            duration //= FMRI.TR
            # hemodynamic_lag = FMRI.HEMODYNAMIC_LAG // FMRI.TR
            # data_items.append((onset+hemodynamic_lag, onset+duration+hemodynamic_lag))
            data_items.append((onset, onset + duration))
            condition, item, key = [row[k] for k in ("condition", "item", "key")]
            item = int(item)
            key = int(key)
            labels.append((condition, item, key))
        data_items = np.array(data_items)
        labels = np.array(labels)
        return data_items, labels

    dataset_path = basedir / "ds000212_learning_from_brains"
    scenarios_csv = basedir / "ds000212_scenarios.csv"
    scenarios = DS000212Scenarios(scenarios_csv, sampling_method)
    assert (
        dataset_path.exists()
    ), f"No dataset found at '{dataset_path} (hint: run.sh datasets)"
    assert scenarios_csv.exists()

    glob_pattern = f"*{subject}*.tar" if subject is not None else "*.tar"
    tarfiles = [str(f) for f in Path(dataset_path).glob(glob_pattern)]
    wdataset = (
        wds.WebDataset(tarfiles, nodesplitter=wds.shardlists.split_by_worker)
        .decode("pil")
        .compose(_get_samples)
    )
    features = Features(
        {
            "label": Sequence(
                feature=Value(dtype="float32"), length=DATA_LENGTH, id=None
            ),
            "input": Value("string", id=None),
            # In the scanner, for each story, participants made moral judgments of
            # the action on a 4-point scale, ‘not at all morally wrong’ (1) and
            # ‘very morally wrong’ (4)
            # https://moralitylab.bc.edu/wp-content/uploads/sites/101/2011/10/ChakroffEtAl_2016_FINAL.pdf
            "behavior": ClassLabel(
                4,
                names=[
                    "not at all morally wrong",
                    "not morally wrong",
                    "morally wrong",
                    "very morally wrong",
                ],
            ),
        }
    )

    def _generator():
        yield from wdataset

    return IterableDataset.from_generator(generator=_generator, features=features)
