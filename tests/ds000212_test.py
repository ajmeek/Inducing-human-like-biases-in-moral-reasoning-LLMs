from torch.utils.data import DataLoader

import data.ds000212.ds000212 as ds000212
import data.ds000212.DS000212Scenarios as DS000212Scenarios
from data.ds000212.constants import Sampling
import pytest


def test_load():
    dss = ds000212.load(
        ds000212.CONFIGURATIONS[0],
        sampling_method=ds000212.Sampling.SENTENCES,
        split="train",
    )
    assert dss
    assert len(dss) == 1
    ds = dss[0]
    for batch in ds:
        input, label = batch["input"], batch["label"]
        assert label
        assert len(label) == 1024
        assert input


@pytest.mark.parametrize("sampling_method", [Sampling.SENTENCES, Sampling.LAST])
def test_DS000212Scenarios(sampling_method):
    scenarios = DS000212Scenarios.DS000212Scenarios(
        "data/ds000212/ds000212_scenarios.csv", sampling_method
    )
    res = scenarios.parse_label("E_NA", 55)
    assert res
    scenario = [
        "You and your partner are on a week-long vacation together. For the first time in a while, you're totally relaxed and not tied to your computer or Blackberry.",
        "You spend much of the week in the hotel room, sleeping or having sex.",
        "This is your first vacation since your honeymoon two years ago.",
        "You didn't realize how much you needed a vacation.",
    ]
    if sampling_method == Sampling.SENTENCES:
        assert len(res) == 4
        for i, e in enumerate(res):
            assert e == " ".join(scenario[: i + 1])
    else:
        assert res == " ".join(scenario)
