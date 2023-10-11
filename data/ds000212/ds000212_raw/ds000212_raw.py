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


from collections import defaultdict
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

"""

_HOMEPAGE = "https://github.com/OpenNeuroDatasets/ds000212"

# TODO: Add the licence.
_LICENSE = ""

_VERSION = datasets.Version("0.9.0")


@dataclass(frozen=True)
class FMRI:
    HEMODYNAMIC_LAG = 6
    TR = 2.0


TRAIN_SPLIT_FACTOR = 0.8
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
            name=f"RAW",  # The sampling is done in Makefile.
            version=_VERSION,
            description=_DESCRIPTION,
        )
    ]

    FMRI_DATA_LENGTH = 39127  # See file 'processed'

    DEFAULT_CONFIG_NAME = BUILDER_CONFIGS[
        0
    ].name  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "label": Sequence(
                    feature=Value(dtype="float32"),
                    length=DS000212.FMRI_DATA_LENGTH,
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
        global participants_list

        assert self.base_path
        base_path = Path(self.base_path)
        assert base_path.exists()

        my_ps = [p for p in participants_list if p.group != "ASD"]
        assert my_ps
        urls = defaultdict(list)
        for i, p in enumerate(my_ps):
            split = (
                str(datasets.Split.TRAIN)
                if i + 1 <= len(my_ps) * TRAIN_SPLIT_FACTOR
                else str(datasets.Split.TEST)
            )
            sub_urls = list(base_path.glob(f"{p.id}/func/*.tar"))
            urls[split].extend(sub_urls)

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
            npz_f = next(archive.glob("**/*.npz"))
            np_file = np.load(npz_f, allow_pickle=True)
            labels = np_file["labels"]
            tr = FMRI.TR
            data_items = np_file["data_items"]

            for idx, (target, label) in enumerate(zip(data_items, labels)):
                target: np.array
                condition, item, behavior_key = label
                assert (
                    behavior_key
                    and behavior_key.isdigit()
                    and (0 <= int(behavior_key) <= BEHAVIOR_KEYS_NUM)
                )
                behavior_key = int(behavior_key)
                if behavior_key == 0:
                    # Silently skip as this. Assume broken data.
                    continue
                sentences = self.scenarios.parse_label(condition, item)
                text = " ".join(sentences)
                assert (
                    len(target) == DS000212.FMRI_DATA_LENGTH
                ), f"{len(target)=} == {DS000212.FMRI_DATA_LENGTH=}"
                error_msg = f"expect each sample has its label but ({len(text)=}, {target.shape=})"
                assert target.ndim == 1 and isinstance(text[0], str), error_msg
                base_key = f"{npz_f}-{idx}"
                yield (
                    base_key,
                    {
                        "label": target,
                        "input": text,
                        "behavior": behavior_key,
                        "file": npz_f,
                    },
                )


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


@dataclass
class Participant:
    id: str
    group: str
    gender: str
    age: int


participants_list = [
    Participant("sub-03", "NT", "M", 22),
    Participant("sub-04", "NT", "F", 19),
    Participant("sub-05", "NT", "M", 20),
    Participant("sub-06", "NT", "F", 20),
    Participant("sub-07", "NT", "F", 19),
    Participant("sub-08", "NT", "M", 26),
    Participant("sub-09", "NT", "F", 22),
    Participant("sub-10", "NT", "M", 25),
    Participant("sub-11", "NT", "F", 24),
    Participant("sub-12", "NT", "M", 23),
    Participant("sub-13", "NT", "M", 22),
    Participant("sub-14", "NT", "M", 23),
    Participant("sub-15", "ASD", "M", 30),
    Participant("sub-16", "ASD", "F", 27),
    Participant("sub-17", "ASD", "M", 38),
    Participant("sub-18", "ASD", "M", 29),
    Participant("sub-19", "ASD", "M", 36),
    Participant("sub-20", "ASD", "M", 46),
    Participant("sub-22", "ASD", "F", 24),
    Participant("sub-23", "ASD", "M", 21),
    Participant("sub-24", "ASD", "M", 27),
    Participant("sub-27", "NT", "F", 39),
    Participant("sub-28", "NT", "F", 32),
    Participant("sub-29", "ASD", "M", 41),
    Participant("sub-30", "ASD", "M", 29),
    Participant("sub-31", "ASD", "M", 36),
    Participant("sub-32", "NT", "M", 50),
    Participant("sub-33", "NT", "M", 40),
    Participant("sub-34", "NT", "F", 33),
    Participant("sub-35", "NT", "M", 45),
    Participant("sub-38", "NT", "M", 43),
    Participant("sub-39", "ASD", "M", 46),
    Participant("sub-40", "NT", "M", 44),
    Participant("sub-41", "NT", "M", 20),
    Participant("sub-42", "NT", "M", 28),
    Participant("sub-44", "ASD", "M", 37),
    Participant("sub-45", "NT", "M", 21),
    Participant("sub-46", "NT", "M", 36),
    Participant("sub-47", "NT", "M", 20),
]
