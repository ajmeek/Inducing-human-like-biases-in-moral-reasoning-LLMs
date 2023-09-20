from .ds000212_lfb import DS000212Scenarios, Sampling
import pytest
from datasets import load_dataset, DownloadMode
from pathlib import Path


@pytest.mark.parametrize("sampling_method", [Sampling.SENTENCES, Sampling.LAST])
def test_load(sampling_method):
    dss = load_dataset(
        path=str(Path(".").absolute()),
        data_dir=str(Path(".").absolute()),
        name=f"LFB-{sampling_method}",
        split=["train", "test"],
        download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    assert dss
    assert len(dss) == 2
    for ds in dss:
        assert ds.num_rows > 0
        for batch in ds:
            input, label, behavior = batch["input"], batch["label"], batch["behavior"]
            assert label
            assert len(label) == 1024
            assert input
            #assert 0 <= behavior <= 3


def test_DS000212Scenarios():
    scenarios = DS000212Scenarios("ds000212_scenarios.csv")
    res = scenarios.parse_label("E_NA", 55)
    assert res
    scenario = [
        "You and your partner are on a week-long vacation together. For the first time in a while, you're totally relaxed and not tied to your computer or Blackberry.",
        "You spend much of the week in the hotel room, sleeping or having sex.",
        "This is your first vacation since your honeymoon two years ago.",
        "You didn't realize how much you needed a vacation.",
    ]
    assert len(res) == 4
    for i, e in enumerate(res):
        assert e == scenario[i]
