from csv import DictReader
from pathlib import Path
from utils.constants import Sampling

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

    def __init__(self, scenarios_csv, config) -> None:
        self._init_scenarios(scenarios_csv)
        self._config = config

    def _init_scenarios(self, scenarios_csv: Path):
        self._scenarios = []
        with open(scenarios_csv, newline='', encoding='utf-8') as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self._scenarios.append(row)

    def parse_label(self, label) -> list[str]:
        condition, item, key = label
        if condition not in DS000212Scenarios.event_to_scenario:
            return None
        skind, stype = DS000212Scenarios.event_to_scenario[condition]
        found = [s for s in self._scenarios if s['item'] == item]
        if not found:
            return None
        found = found[0]
        assert found['type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {event}."

        # TODO: clean up later
        # For now we do multiple
        part1 = ' '.join([
            found['background'],
        ])
        part2 = ' '.join([
            found['background'],
            found['action'],
        ])
        part3 = ' '.join([
            found['background'],
            found['action'],
            found['outcome'],
        ])
        part4 = ' '.join([
            found['background'],
            found['action'],
            found['outcome'],
            found[skind]
        ])
        if self._config['sampling_method'] in Sampling.ONE_POINT_METHODS:
            len_intervals = 1
        elif self._config['sampling_method'] is Sampling.SENTENCES:
            len_intervals = 4
        else:
            raise NotImplementedError()
        if len_intervals == 1:
            return part4
        elif len_intervals == 2:
            return [part2, part4]
        elif len_intervals == 4:
            return [part1, part2, part3, part4]
        else:
            raise ValueError(f"Unexpected length of intervals: {len_intervals}")