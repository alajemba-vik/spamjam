import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Loads, downloads, and converts the Spambase dataset with robust error handling."""

    def __init__(self, data_dir: str = "../data/raw"):
        self.data_dir = Path(data_dir)
        self.raw_path = self.data_dir / "spambase.data"
        self.csv_path = self.data_dir / "spambase.csv"
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

        # The 58 official column names (57 features + 1 target)
        self.columns = [
            'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
            'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
            'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
            'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
            'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
            'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
            'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
            'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
            'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
            'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
            'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
            'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
            'char_freq_semicolon', 'char_freq_parenthesis', 'char_freq_bracket',
            'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash',
            'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
            'is_spam'
        ]

    def load_data(self) -> pd.DataFrame:
        """
        1. Loads existing CSV (fastest).
        2. Converts local .data to CSV.
        3. Downloads .data, then converts to CSV.
        """
        try:
            # 1. Load CSV directly if it already exists
            if self.csv_path.exists():
                print(f" Found existing CSV: {self.csv_path}")
                return pd.read_csv(self.csv_path)

            # 2. Download raw data if it doesn't exist
            if not self.raw_path.exists():
                print(" No local data found. Downloading from UCI...")
                self.data_dir.mkdir(parents=True, exist_ok=True)
                download_path = self.raw_path.with_suffix(".download")
                urllib.request.urlretrieve(self.url, download_path)
                download_path.replace(self.raw_path)
                print(" Download complete.")

            # 3. Convert .data to DataFrame and save as CSV
            print(f" Converting raw file to CSV at {self.csv_path}...")
            df = pd.read_csv(self.raw_path, header=None, names=self.columns)
            df.to_csv(self.csv_path, index=False)
            print(" Conversion successful.")

            return df

        except urllib.error.URLError as e:
            raise ConnectionError(f" Network error: Could not download the data. Details: {e}")
        except Exception as e:
            raise RuntimeError(f" Error processing the data: {e}")

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Checks for and removes duplicate rows to prevent data leakage."""
        # 1. Check
        duplicates_count = df.duplicated().sum()
        print(f" Found {duplicates_count} duplicate rows.")

        # 2. Delete (if any exist)
        if duplicates_count > 0:
            # reset_index(drop=True) ensures the row numbers stay clean
            df_clean = df.drop_duplicates().reset_index(drop=True)
            print(f" Duplicates removed. New shape: {df_clean.shape}")
            return df_clean
        else:
            print(" No duplicates found.")
            return df

    def split_and_save(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        processed_dir = self.data_dir.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        X = df.drop(columns=['is_spam'])
        y = df['is_spam']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(processed_dir / "sb_train.csv", index=False)
        test_df.to_csv(processed_dir / "sb_test.csv", index=False)

        print(f" Saved sb_train.csv ({train_df.shape[0]} rows)")
        print(f" Saved sb_test.csv ({test_df.shape[0]} rows)")

        return train_df, test_df