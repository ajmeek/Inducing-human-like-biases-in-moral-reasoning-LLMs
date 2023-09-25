# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" HuggingFace datasets build script for the ds000212 dataset. """


from csv import DictReader
from dataclasses import dataclass
from datasets import Value, Sequence, ClassLabel
from pathlib import Path
from re import search
from typing import Union, List
import datasets
import numpy as np

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

_DESCRIPTION = """\
Description: 
This dataset contains data on two tasks:
- dis: moral judgments on a 1-4 scale of moral-violation scenarios
	- Scenario breakdown:
		- Items 1-12: Physical harms (e.g., stabbing)
		- Items 13-24: Psychological harms (e.g., insults)
		- Items 25-36: Incest violations (e.g., sleeping with a sibling)
		- Items 37-48: Pathogen violations (e.g., drinking human blood)
		- Items 49-60: Neutral scenarios
	- Condition labels:
		- A_PHA: Accidental physical harms
		- B_PSA: Accidental psychological harms
		- C_IA: Accidental incest violations
		- D_PA: Accidental pathogen violations
		- E_NA: Accidental neutral scenarios
		- F_PHI: Intentional physical harms
		- G_PSI: Intentional psychological harms
		- H_II: Intentional incest violations
		- I_PI: Intentional pathogen violations
		- J_NI: Intentional neutral scenarios
- tom/tomloc/fb: Theory of Mind (ToM) localizer task, including false-belief and false-photo trials (Saxe & Kanwisher, 2003).

Participants:
Participants include neurotypical (NT) participants and participants with Autism Spectrum Disorder (ASD).

Data collection:
All data was collected in 2011 at the Massachusetts Institute of Technology.

Papers containing analyses from this data:
- Chakroff et al. (2016). When minds matter for moral judgment: intent information is neurally encoded for harmful but not impure acts. Social Cognitive & Affective Neuroscience. doi: 10.1093/scan/nsv131
- Koster-Hale et al. (2013).  Decoding moral judgments from neural representations of intentions. Proceedings of the National Academy of Sciences. doi: 10.1073/pnas.1207992110
- Wasserman et al. (in prep). Illuminating the conceptual structure of the space of moral violations with searchlight representational similarity analysis.


------------------ 

https://github.com/athms/learning-from-brains
Data and models from the paper: Thomas, A., Ré, C., & Poldrack, R. (2022). Self-supervised learning of brain dynamics from broad neuroimaging data. Advances in Neural Information Processing Systems, 35, 21255-21269.

"""

_HOMEPAGE = "https://github.com/athms/learning-from-brains"

# TODO: Add the licence.
# See https://github.com/athms/learning-from-brains/issues/5
_LICENSE = ""

_VERSION = datasets.Version("0.9.0")


@dataclass(frozen=True)
class FMRI:
    HEMODYNAMIC_LAG = 6
    TR = 2.0

BEHAVIOR_KEYS_NUM = 4

@dataclass(frozen=True)
class Periods:
    BACKGROUND = 6
    ACTION = 4
    OUTCOME = 4
    INTENT = 4
    JUDGMENT = 4
    ENDS = [BACKGROUND, ACTION, OUTCOME, INTENT, JUDGMENT]


@dataclass(frozen=True)
class Sampling:
    LAST = "LAST"
    AVG = "AVG"
    MIDDLE = "MIDDLE"
    SENTENCES = "SENTENCES"
    ONE_POINT_METHODS = [LAST, AVG, MIDDLE]
    ALL = [LAST, AVG, MIDDLE, SENTENCES]


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class DS000212(datasets.GeneratorBasedBuilder):
    """The ds000212 dataset from OpenNeuro."""

    VERSION = _VERSION

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=f"LFB-{sm}",
            version=_VERSION,
            description="The ds000212 dataset preprocessed by Thomas, A. W., Ré, C., & Poldrack, R. A. (2022)."
            "Self-Supervised Learning Of Brain Dynamics From Broad Neuroimaging Data. arXiv preprint arXiv:2206.11417.",
        )
        for sm in (Sampling.ALL)
    ]

    LFB_DATA_LENGTH = 1024

    DEFAULT_CONFIG_NAME = "LFB"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "label": Sequence(
                    feature=Value(dtype="float32"),
                    length=DS000212.LFB_DATA_LENGTH,
                    id=None,
                ),
                "input": Value("string", id=None),
                "file": Value("string", id=None),
                "behavior": ClassLabel(
                    BEHAVIOR_KEYS_NUM + 1,
                    names=[
                        "NA",
                        "not at all morally wrong",
                        "not morally wrong",
                        "morally wrong",
                        "very morally wrong",
                    ],
                ),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Called by HuggingFace to generate splits."""

        self.sampling_method = self.config.name.split("-")[1]
        assert self.base_path
        base_path = Path(self.base_path)
        assert base_path.exists()
        urls = {
            split: [
                str(p)
                for ext in ("tar", "tsv")
                for p in Path(base_path / split).glob(f"*.{ext}")
            ]
            for split in ("train", "test")
        }
        urls["scenarios_file"] = str(Path(base_path / "ds000212_scenarios.csv"))
        dl_result = dl_manager.extract(urls)
        self.scenarios = DS000212Scenarios(dl_result["scenarios_file"])
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"files": dl_result[str(split)], "split": str(split)},
            )
            for split in (datasets.Split.TRAIN, datasets.Split.TEST)
        ]

    def _generate_examples(self, files, split):
        """Called by HuggingFace to generate examples."""

        files = [Path(f) for f in files]

        for archive in (d for d in files if d.is_dir()):
            bold_f = next(archive.glob("*.bold.pyd"))
            tr_f = next(archive.glob("*.t_r.pyd"), None)
            tsv_f = next(archive.glob("*.tsv"), None)
            if not tsv_f:
                continue
            if tr_f and tr_f.exists():
                tr = np.load(tr_f, allow_pickle=True).item()
            else:
                tr = FMRI.TR
            assert int(tr) == tr
            tr = int(tr)
            bold = np.load(bold_f, allow_pickle=True)
            assert tsv_f.exists()
            data_items, labels = self._process_tsv(tsv_f, tr)
            last_end = data_items[-1][-1]
            if bold.shape[0] <= last_end:
                # Skip this run. Assume broken data.
                # Examples: ./ds-ds000212_sub-07_task-dis_run-2.tar
                continue
            for (start, end), label in zip(data_items, labels):
                s_bold = bold[start:end]  # Scenario fMRI data in BOLD.
                base_key = f"{bold_f}-{start}"
                yield from self._generate_from_scenario(base_key, s_bold, tr, label)

    def _generate_from_scenario(self, base_key, s_bold, tr: int, label):
        condition, item, behavior_key = label

        sentences = self.scenarios.parse_label(condition, item)
        if not sentences:
            # Silently skip as this might relate to 'fb' (false beliefs)
            return
        assert behavior_key and behavior_key.isdigit() and (0 <= int(behavior_key) <= BEHAVIOR_KEYS_NUM)
        behavior_key = int(behavior_key)

        if self.sampling_method == Sampling.SENTENCES:
            text = sentences
        else:
            text = " ".join(sentences)
        target: np.array = self._sample_from_bold_sequence(
            s_bold, self.sampling_method, tr
        )
        error_msg = (
            f"expect each sample has its label but ({len(text)=}, {target.shape=})"
        )
        if target.ndim == 1:
            assert (
                len(target) == DS000212.LFB_DATA_LENGTH
            ), f"{len(target)=} == {DS000212.LFB_DATA_LENGTH=}"
            assert target.ndim == 1 and isinstance(text[0], str), error_msg
            yield (
                base_key,
                {
                    "label": target,
                    "input": text,
                    "behavior": behavior_key,
                    "file": base_key
                },
            )
        else:
            assert target.shape[0] == len(text), error_msg
            for i in range(target.shape[0]):
                assert (
                    len(target[i]) == DS000212.LFB_DATA_LENGTH
                ), f"{len(target[i])=} == {DS000212.LFB_DATA_LENGTH=}"
                yield (
                    f"{base_key}-{i}",
                    {
                        "label": target[i],
                        "input": text[i],
                        "behavior": behavior_key,
                        "file": base_key
                    },
                )

    def _sample_from_bold_sequence(
        self, sequence: np.array, method: Sampling, tr: int
    ) -> np.array:
        # TODO: To check with sources. There might be a bug with subtracting 
        # the hemodynamic lag because those 'onset' and 'duration' fields add 
        # up with 161 as the last point. And there are total 166 points in bold sequence.
        # Meaning onset+duration points to correct point? Or onset + duration + h. lag?
        if method == Sampling.LAST:
            result = sequence[-(FMRI.HEMODYNAMIC_LAG // tr)]
        elif method == Sampling.AVG:
            result = np.average(sequence, axis=-2)
        elif method == Sampling.MIDDLE:
            result = sequence[len(sequence) // 2]
        elif method == Sampling.SENTENCES:
            result = sequence[
                [p // tr for p in Periods.ENDS[:3] + [-FMRI.HEMODYNAMIC_LAG]]
            ]
        else:
            raise NotImplementedError()
        return result

    def _process_tsv(self, from_tsv: Path, tr: float):
        rows = []
        with open(from_tsv, newline="") as csvfile:
            rows = list(DictReader(csvfile, delimiter="\t", quotechar='"'))
        data_items = []
        labels = []
        for row in rows:
            onset = row["onset"]
            try:
                if "[" in onset and "]" in onset:
                    m = search(r"\d+", onset)
                    assert m
                    onset = int(m.group(0))
                else:
                    onset = int(onset)
            except Exception as e:
                print(f'Failed to parse "onset" for {from_tsv}: {e}')
                continue
            duration = int(row["duration"])
            onset = int(onset // tr)
            duration = int(duration // tr)
            data_items.append((onset, onset + duration))
            condition, item, key = [row[k] for k in ("condition", "item", "key")]
            item = int(item)
            key = int(key)
            labels.append((condition, item, key))
        data_items = np.array(data_items)
        labels = np.array(labels)
        return data_items, labels


class DS000212Scenarios(object):
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

    def __init__(self, scenarios_csv: Union[str, Path]) -> None:
        assert scenarios_csv
        with open(scenarios_csv, newline="", encoding="utf-8") as csvfile:
            reader = DictReader(csvfile)
            self._scenarios = [row for row in reader]

    def parse_label(self, condition: str, item: int) -> List[str]:
        if condition not in DS000212Scenarios.event_to_scenario:
            return None
        a_or_i, stype = DS000212Scenarios.event_to_scenario[condition]
        assert a_or_i in ("accidental", "intentional")
        found = next((s for s in self._scenarios if s["item"] == str(item)), None)
        if not found:
            return None
        assert (
            found["type"] == stype
        ), f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}."

        part_names = ["background", "action", "outcome", a_or_i]
        return [found[p] for p in part_names]
