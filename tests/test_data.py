"""Tests for data validation, cleaning and splitting."""

import pandas as pd
import pytest

from spamjam import constants

SPAMBASE_SHAPE = (4601, 58)   # emails, columns (57 features + target), from UCI
STRATIFY_TOLERANCE = 0.02     # how far the spam share may move after the split

# ... as documented by UCI.
def test_dataset_has_expected_shape(real_df):
    assert real_df.shape == SPAMBASE_SHAPE
    assert constants.TARGET_COL in real_df.columns

# 0 (ham) and 1 (spam)
def test_target_is_binary(real_df):
    labels = set(real_df[constants.TARGET_COL].unique())
    assert labels == {constants.HAM, constants.SPAM}

def test_no_missing_or_negative_values(real_df):
    assert real_df.isna().sum().sum() == 0
    assert (real_df.drop(columns=constants.TARGET_COL) >= 0).all().all()

def test_remove_duplicates_drops_exact_copies(loader, small_df):
    with_copies = pd.concat([small_df, small_df.head(5)], ignore_index=True)
    clean = loader.remove_duplicates(with_copies)
    assert not clean.duplicated().any()
    assert len(clean) == len(small_df.drop_duplicates())

def test_remove_duplicates_keeps_columns(loader, small_df):
    clean = loader.remove_duplicates(small_df)
    assert list(clean.columns) == list(small_df.columns)

# --- Splitting ---------------------------------------------------------------

def test_no_test_row_in_train(loader, small_df):
    df = loader.remove_duplicates(small_df)
    train, test = loader.split(df)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(df)

def test_split_is_stratified(loader, small_df):
    train, test = loader.split(small_df)
    spam_ratio = small_df[constants.TARGET_COL].mean()
    assert train[constants.TARGET_COL].mean() == pytest.approx(
        spam_ratio, abs=STRATIFY_TOLERANCE
    )
    assert test[constants.TARGET_COL].mean() == pytest.approx(
        spam_ratio, abs=STRATIFY_TOLERANCE
    )

def test_split_is_reproducible(loader, small_df):
    train_a, _ = loader.split(small_df)
    train_b, _ = loader.split(small_df)
    assert train_a.index.equals(train_b.index)

def test_save_and_load_split_round_trip(loader, small_df):
    train, test = loader.split(small_df)
    loader.save_split(train, test)
    train_back, test_back = loader.load_split()
    assert train_back.shape == train.shape
    assert test_back.shape == test.shape


def test_split_is_created_once_then_loaded(loader, small_df):
    small_df.to_csv(loader.data_path, index=False)  # no download needed
    train_df, test_df = loader.load_or_create_split()
    assert loader.train_path.exists() and loader.test_path.exists()

    # Second call: the saved files are loaded, not a new split.
    train_again, test_again = loader.load_or_create_split()
    assert train_again.equals(pd.read_csv(loader.train_path))
    assert len(train_again) == len(train_df) and len(test_again) == len(test_df)
