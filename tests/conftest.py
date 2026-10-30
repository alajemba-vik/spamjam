"""Shared pytest fixtures, built from our real dataset."""

import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split

from spamjam import constants
from spamjam.cli import main
from spamjam.config import load_config
from spamjam.constants import PROJECT_ROOT, TARGET_COL
from spamjam.data import DatasetLoader

TMP_DATA_FILE = "spambase.csv"
TMP_TRAIN_FILE = "train.csv"
TMP_TEST_FILE = "test.csv"
TMP_CONFIG_FILE = "config.yaml"
TMP_MODEL_FILE = "model.joblib"
TMP_MODEL_NO_DONOR_FILE = "model_without_donor_words.joblib"

# Use a random sample of 300 real emails instead of all say 4601, so the
# tests run faster, and they check that our CODE works, which doesn't need all
# the data
SAMPLE_SIZE = 300
SAMPLE_SEED = 0
# Emails used to train the models of the command-line and Streamlit tests,
# with a one-value grid so training is fast.
TRAINED_SAMPLE_SIZE = 600
FAST_GRID = {"model__C": [1]}

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


@pytest.fixture
def train_test(loader, small_df):
    """The sample split into X_train, y_train, X_test, y_test."""
    train, test = loader.split(small_df)
    target = constants.TARGET_COL
    return (
        train.drop(columns=target), train[target],
        test.drop(columns=target), test[target],
    )


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """A copy of config.yaml where everything happens in a temporary folder,
    and both versions of the model trained with it.

    The tests never need `spamjam train` to have been run first: on a fresh
    clone, pytest comes before training.

    The real test set must only be used once, at the end of the project, so
    the tests use their own small train/test split from a sample of the data.
    The grid is reduced to one value so training is fast.
    """
    config = load_config()
    data_path = PROJECT_ROOT / config["data"]["path"]
    if not data_path.exists():
        pytest.skip("the dataset is missing")
    folder = tmp_path_factory.mktemp("cli")

    sample = pd.read_csv(data_path).sample(n=TRAINED_SAMPLE_SIZE, random_state=0)
    train_df, test_df = train_test_split(
        sample, test_size=0.2, stratify=sample[TARGET_COL], random_state=0
    )
    train_df.to_csv(folder / TMP_TRAIN_FILE, index=False)
    test_df.to_csv(folder / TMP_TEST_FILE, index=False)

    config["data"]["train_path"] = str(folder / TMP_TRAIN_FILE)
    config["data"]["test_path"] = str(folder / TMP_TEST_FILE)
    config["model"]["path"] = str(folder / TMP_MODEL_FILE)
    config["model"]["without_donor_words"]["path"] = str(
        folder / TMP_MODEL_NO_DONOR_FILE
    )
    config["grids"][config["model"]["name"]] = FAST_GRID
    config_path = folder / TMP_CONFIG_FILE
    config_path.write_text(yaml.safe_dump(config))

    main(["--config", str(config_path), "train"])
    return config_path, folder
