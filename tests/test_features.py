"""Tests for feature engineering and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from spamjam import constants
from spamjam.features import build_pipeline


@pytest.fixture
def train_test(loader, small_df):
    train, test = loader.split(small_df)
    return (
        train.drop(columns=constants.TARGET_COL), train[constants.TARGET_COL],
        test.drop(columns=constants.TARGET_COL), test[constants.TARGET_COL],
    )

# We fit the pipeline on the training rows only. Then we check that
# the scaler's stored means are EQUAL to the training means, and
# DIFFERENT from the means of train + test together.
# If the scaler had seen test rows, the first check would fail.
def test_scalers_are_fitted_on_train_only(train_test):
    X_train, y_train, X_test, _ = train_test
    pipe = build_pipeline(LogisticRegression()).fit(X_train, y_train)
    parts = pipe["preprocess"].named_transformers_
    all_rows = pd.concat([X_train, X_test])

    # (word and character frequencies): the scaler sees the raw values.
    scaler = parts["other"]["scale"]
    cols = scaler.feature_names_in_
    assert np.allclose(scaler.mean_, X_train[cols].mean())
    assert not np.allclose(scaler.mean_, all_rows[cols].mean())

    # (capital-letter counts): the scaler sees log(1 + x), so compare
    # with the mean of the log-transformed training values.
    scaler = parts["capital"]["scale"]
    cols = scaler.feature_names_in_
    assert np.allclose(scaler.mean_, np.log1p(X_train[cols]).mean())
    assert not np.allclose(scaler.mean_, np.log1p(all_rows[cols]).mean())

def test_pipeline_outputs_probabilities(train_test):
    X_train, y_train, X_test, _ = train_test
    pipe = build_pipeline(LogisticRegression()).fit(X_train, y_train)
    # predict_proba gives 2 columns (ham, spam); [:, 1] keeps the spam column.
    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()
