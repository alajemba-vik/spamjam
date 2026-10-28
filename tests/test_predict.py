"""Tests for predicting one new email."""

import json

import pytest

from spamjam import constants
from spamjam.models import Trainer
from spamjam.predict import predict_email, read_email_features


@pytest.fixture
def model_and_features(loader, small_df):
    train, test = loader.split(small_df)
    trainer = Trainer("logistic_regression", {}, n_splits=3,
                      scoring="average_precision", random_state=0)
    model = trainer.train(
        train.drop(columns=constants.TARGET_COL), train[constants.TARGET_COL]
    )
    features = test.drop(columns=constants.TARGET_COL).iloc[0].to_dict()
    return model, features


def test_prediction_is_a_probability_and_a_label(model_and_features):
    model, features = model_and_features
    result = predict_email(model, features, threshold=0.5)
    assert 0 <= result["spam_probability"] <= 1
    assert result["is_spam"] == int(result["spam_probability"] >= 0.5)


def test_missing_value_is_rejected(model_and_features):
    model, features = model_and_features
    features.pop("word_freq_free")
    with pytest.raises(ValueError, match="Missing values"):
        predict_email(model, features, threshold=0.5)


def test_unknown_feature_is_rejected(model_and_features):
    model, features = model_and_features
    features["word_freq_unknown"] = 0.0
    with pytest.raises(ValueError, match="Unknown features"):
        predict_email(model, features, threshold=0.5)


def test_non_number_is_rejected(model_and_features):
    model, features = model_and_features
    features["word_freq_free"] = "a lot"
    with pytest.raises(ValueError, match="must be numbers"):
        predict_email(model, features, threshold=0.5)


def test_infinite_value_is_rejected(model_and_features):
    model, features = model_and_features
    features["word_freq_free"] = float("inf")
    with pytest.raises(ValueError, match="must be numbers"):
        predict_email(model, features, threshold=0.5)


def test_read_email_features_from_json(model_and_features, tmp_path):
    _, features = model_and_features
    path = tmp_path / "new_email.json"
    path.write_text(json.dumps(features))
    assert read_email_features(path) == features

    path.write_text(json.dumps([features]))
    with pytest.raises(ValueError, match="one JSON object"):
        read_email_features(path)
