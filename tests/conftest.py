"""Shared pytest fixtures, built from our real dataset."""

import pandas as pd
import pytest

from spamjam.data import DatasetLoader

TMP_DATA_FILE = "spambase.csv"
TMP_TRAIN_FILE = "train.csv"
TMP_TEST_FILE = "test.csv"

# Use a random sample of 300 real emails instead of all say 4601, so the
# tests run faster, and they check that our CODE works, which doesn't need all
# the data
SAMPLE_SIZE = 300
SAMPLE_SEED = 0

@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    """The full Spambase dataset, read once from the path in config.yaml."""
    loader = DatasetLoader.from_config()
    if not loader.data_path.exists():
        pytest.skip(f"{loader.data_path} is missing")
    return loader.load_data()

@pytest.fixture
def small_df(real_df: pd.DataFrame) -> pd.DataFrame:
    """A fresh random sample of real emails for each test, so tests stay fast."""
    return real_df.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED).reset_index(
        drop=True
    )

@pytest.fixture
def loader(tmp_path) -> DatasetLoader:
    """A loader with our config settings, but writing to a temporary folder."""
    config_loader = DatasetLoader.from_config()
    return DatasetLoader(
        data_path=tmp_path / TMP_DATA_FILE,
        uci_id=config_loader.uci_id,
        test_size=config_loader.test_size,
        random_state=config_loader.random_state,
        train_path=tmp_path / TMP_TRAIN_FILE,
        test_path=tmp_path / TMP_TEST_FILE,
    )
