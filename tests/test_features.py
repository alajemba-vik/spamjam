"""Tests for feature engineering and the preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
