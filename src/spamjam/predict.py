"""Predict whether one new email is spam.

Our model never sees the text of an email. Like the Spambase data it was
trained on, it only sees 57 numbers per email: how often some words and
characters appear, and how long the runs of capital letters are. So a "new
email" for the model is those 57 named values. `spamjam.text` computes them
from the text of an email.

We read them from a JSON file, because JSON stores each value next to its
feature name, so the file is easy to read and the order doesn't matter::

    {"word_freq_free": 0.32, "word_freq_money": 0.0, ...}

The values are checked before predicting: every feature the model was trained
on must be present, known and a number. We reject bad input instead of
guessing missing values, so a typo can't silently change the prediction.
"""

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from spamjam.constants import HAM, SPAM


def read_email_features(path: Path) -> dict:
    """Read one email's feature values from a JSON file."""
    with open(path, encoding="utf-8") as f:
        features = json.load(f)
    if not isinstance(features, dict):
        raise ValueError("The input must be one JSON object: {feature: value, ...}.")
    return features


def features_to_row(features: dict, feature_names: list[str]) -> pd.DataFrame:
    """Check one email's feature values and return them as a one-row table."""
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"Missing values for: {', '.join(missing)}")

    unknown = [name for name in features if name not in feature_names]
    if unknown:
        raise ValueError(f"Unknown features: {', '.join(unknown)}")

    not_numbers = [
        name
        for name in feature_names
        if not isinstance(features[name], (int, float))
        or not math.isfinite(features[name])
    ]
    if not_numbers:
        raise ValueError(f"Values must be numbers: {', '.join(not_numbers)}")

    # Same columns, in the same order, as the data the model was trained on.
    return pd.DataFrame([features], columns=feature_names)


def predict_email(model: Pipeline, features: dict, threshold: float) -> dict:
    """Return the spam probability and the label (SPAM or HAM) for one email."""
    row = features_to_row(features, list(model.feature_names_in_))
    spam_probability = model.predict_proba(row)[0, SPAM]  # first row, spam column
    return {
        "spam_probability": float(spam_probability),
        "is_spam": SPAM if spam_probability >= threshold else HAM,
    }
