import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

import spamjam.constants as constants
from spamjam.config import load_config

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load the dataset, remove duplicates and split it into train and test sets."""

    def __init__(self, data_path: Path, uci_id: int,
                 test_size: float, random_state: int,
                 train_path: Path, test_path: Path) -> None:
        self.data_path = data_path
        self.uci_id = uci_id
        self.test_size = test_size
        self.random_state = random_state
        self.train_path = train_path
        self.test_path = test_path

    @classmethod
    def from_config(cls, config_path: Path = constants.CONFIG_PATH) -> "DatasetLoader":
        """Create a loader from the settings in config.yaml"""
        config = load_config(config_path)
        return cls(
            data_path=constants.PROJECT_ROOT/config["data"]["path"],
            uci_id=config["data"]["uci_id"],
            test_size=config["split"]["test_size"],
            random_state=config["random_state"],
            train_path=constants.PROJECT_ROOT/config["data"]["train_path"],
            test_path=constants.PROJECT_ROOT/config["data"]["test_path"]
        )

    def load_data(self) -> pd.DataFrame:
        """Download the dataset, save on the first run and return it."""
        if self.data_path.exists():
            logger.info("Reading %s", self.data_path)
            return pd.read_csv(self.data_path)

        logger.info("Downloading UCI dataset %d", self.uci_id)
        dataset = fetch_ucirepo(id=self.uci_id)
        df = dataset.data.features.copy()
        # Get all rows, first column
        df[constants.TARGET_COL] = dataset.data.targets.iloc[:, 0]

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_path, index=False)
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the data without duplicate rows,
        so no email can be in both train and test."""
        clean = df.drop_duplicates().reset_index(drop=True)
        logger.info("Removed %d duplicate rows", len(df) - len(clean))
        return clean

    def save_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save the train and test sets to the paths in config.yaml."""
        self.train_path.parent.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)

        logger.info("Saved %s and %s", self.train_path, self.test_path)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into stratified train and test sets."""
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df[constants.TARGET_COL],
            random_state=self.random_state,
        )
        return train_df, test_df

    def load_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the train and test sets from disk."""
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        return train_df, test_df
