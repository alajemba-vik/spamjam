"""Tests for data validation, cleaning and splitting."""

import pandas as pd
import pytest

from spamjam import constants

TARGET = constants.TARGET_COL

# ... as documented by UCI.
def test_dataset_has_expected_shape(real_df):
    assert real_df.shape == (4601, 58)
    assert TARGET in real_df.columns

# 0 (ham) and 1 (spam)
def test_target_is_binary(real_df):
    assert set(real_df[TARGET].unique()) == {0, 1}

def test_no_missing_or_negative_values(real_df):
    assert real_df.isna().sum().sum() == 0
    assert (real_df.drop(columns=TARGET) >= 0).all().all()

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
    assert train[TARGET].mean() == pytest.approx(small_df[TARGET].mean(), abs=0.02)
    assert test[TARGET].mean() == pytest.approx(small_df[TARGET].mean(), abs=0.02)

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
