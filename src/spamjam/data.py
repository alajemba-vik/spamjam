from urllib.request import urlretrieve
from urllib.error import URLError
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import tempfile
import os

#import constants from our separate file
from spamjam.constants import COLUMNS, TARGET_COL

class DatasetLoader:
    """Loads, downloads, and converts the Spambase dataset with robust error handling."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config=yaml.safe_load(f)

        self.url=config["data"]["url"]
        self.test_size=config["modeling"]["test_size"]
        self.random_state=config["modeling"]["random_state"]


    def load_data(self) -> pd.DataFrame:
        """
        Download the data to a temp file, reads it and deletes the file
        """
        try:
            # 1. Load CSV directly if it already exists
            temp_fd, temp_path = tempfile.mkstemp()
            os.close(temp_fd)

            print("Downloading data...")
            urlretrieve(self.url, temp_path)

            # 2. Download raw data if it doesn't exist
            df = pd.read_csv(temp_path, header=None, names=COLUMNS)

            # 3. Convert .data to DataFrame and save as CSV
            os.remove(temp_path)
            print("Data loaded into memory successfully.")

            return df

        except URLError as e:
            raise ConnectionError(f" Network error: Could not download the data: {e}")
        except Exception as e:
            raise RuntimeError(f" Error processing the data: {e}")

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates."""
        if df.duplicated().sum() > 0:
            return df.drop_duplicates().reset_index(drop=True)
        return df

    def split_and_save(self, df: pd.DataFrame):
        """Splits into train/test DataFrames in memory (no saving)."""
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        return train_df, test_df