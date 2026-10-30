"""Tests for predicting one new email."""

import json
import re

import pytest

from spamjam import constants
from spamjam.evaluate import AVERAGE_PRECISION
from spamjam.models import LOGISTIC_REGRESSION, Trainer
from spamjam.predict import (
    ERROR_MISSING,
    ERROR_NOT_NUMBERS,
    ERROR_NOT_OBJECT,
    ERROR_UNKNOWN,
    SPAM_PROBABILITY,
    parse_features,
    predict_email,
    read_email_features,
)

THRESHOLD = 0.5


@pytest.fixture
def model_and_features(loader, small_df):
    train, test = loader.split(small_df)
    trainer = Trainer(LOGISTIC_REGRESSION, {}, n_splits=3,
                      scoring=AVERAGE_PRECISION, random_state=0)
    model = trainer.train(
        train.drop(columns=constants.TARGET_COL), train[constants.TARGET_COL]
    )
    features = test.drop(columns=constants.TARGET_COL).iloc[0].to_dict()
    return model, features


def test_prediction_is_a_probability_and_a_label(model_and_features):
    model, features = model_and_features
    result = predict_email(model, features, threshold=THRESHOLD)
    assert 0 <= result[SPAM_PROBABILITY] <= 1
    assert result[constants.TARGET_COL] == int(result[SPAM_PROBABILITY] >= THRESHOLD)


def test_missing_value_is_rejected(model_and_features):
    model, features = model_and_features
    features.pop("word_freq_free")
    with pytest.raises(ValueError, match=ERROR_MISSING.format(names="word_freq_free")):
        predict_email(model, features, threshold=THRESHOLD)


def test_unknown_feature_is_rejected(model_and_features):
    model, features = model_and_features
    features["word_freq_unknown"] = 0.0
    error = ERROR_UNKNOWN.format(names="word_freq_unknown")
    with pytest.raises(ValueError, match=error):
        predict_email(model, features, threshold=THRESHOLD)


def test_non_number_is_rejected(model_and_features):
    model, features = model_and_features
    features["word_freq_free"] = "a lot"
    error = ERROR_NOT_NUMBERS.format(names="word_freq_free")
    with pytest.raises(ValueError, match=error):
        predict_email(model, features, threshold=THRESHOLD)


def test_infinite_value_is_rejected(model_and_features):
    model, features = model_and_features
    features["word_freq_free"] = float("inf")
    error = ERROR_NOT_NUMBERS.format(names="word_freq_free")
    with pytest.raises(ValueError, match=error):
        predict_email(model, features, threshold=THRESHOLD)


def test_read_email_features_from_json(model_and_features, tmp_path):
    _, features = model_and_features
    path = tmp_path / "new_email.json"
    path.write_text(json.dumps(features))
    assert read_email_features(path) == features

    path.write_text(json.dumps([features]))
    with pytest.raises(ValueError, match=re.escape(ERROR_NOT_OBJECT)):
        read_email_features(path)


def test_parse_features_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_features("not json")
