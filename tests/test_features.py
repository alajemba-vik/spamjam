"""Tests for feature engineering and the preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from spamjam.config import load_config
from spamjam.features import build_pipeline


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


def test_second_version_drops_the_donor_words_but_takes_all_features(train_test):
    X_train, y_train, _, _ = train_test
    donor = load_config()["features"]["donor_columns"]
    pipeline = build_pipeline(LogisticRegression(), drop_columns=donor)
    pipeline.fit(X_train, y_train)
    assert len(pipeline.feature_names_in_) == X_train.shape[1]   # all 57 in
    used = pipeline["preprocess"].get_feature_names_out()
    assert not any(name.endswith(tuple(donor)) for name in used)
    assert len(used) == X_train.shape[1] - len(donor)
