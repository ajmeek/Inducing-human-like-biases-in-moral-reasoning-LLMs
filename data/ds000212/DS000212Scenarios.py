from csv import DictReader
from pathlib import Path
from typing import List
from .constants import Sampling


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

    def __init__(self, scenarios_csv, sampling: Sampling) -> None:
        self._init_scenarios(scenarios_csv)
        self._sampling = sampling

    def _init_scenarios(self, scenarios_csv: Path):
        self._scenarios = []
        with open(scenarios_csv, newline="", encoding="utf-8") as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self._scenarios.append(row)

    def parse_label(self, condition : str, item : int) -> List[str]:
        if condition not in DS000212Scenarios.event_to_scenario:
            return None
        a_or_i, stype = DS000212Scenarios.event_to_scenario[condition]
        assert a_or_i in ('accidental', 'intentional')
        found = next((s for s in self._scenarios if s["item"] == str(item)), None)
        if not found:
            return None
        assert (
            found["type"] == stype
        ), f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}."

        part_names = ["background", "action", "outcome", a_or_i]
        if self._sampling in Sampling.ONE_POINT_METHODS:
            return " ".join(found[k] for k in part_names)
        elif self._sampling == Sampling.SENTENCES:
            return [" ".join(found[k] for k in part_names[:num]) for num in range(1, 5)]
        else:
            raise NotImplementedError()
