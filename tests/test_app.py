"""Tests for the Streamlit page: input parsing, explanations, and that it runs."""

import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from spamjam.app import BUTTON_CHECK, feature_contributions, parse_features
from spamjam.constants import PROJECT_ROOT
from spamjam.models import Trainer


def test_parse_features_reads_a_json_object():
    features = {"word_freq_free": 0.5, "char_freq_$": 0.1}
    assert parse_features(json.dumps(features)) == features


def test_parse_features_rejects_a_list():
    with pytest.raises(ValueError, match="one JSON object"):
        parse_features("[1, 2, 3]")


def test_parse_features_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_features("not json")


@pytest.mark.parametrize("drop_columns", [None, ["word_freq_george", "word_freq_hp"]])
def test_contributions_add_up_to_the_model_score(train_test, drop_columns):
    X_train, y_train, X_test, _ = train_test
    trainer = Trainer("logistic_regression", {}, n_splits=3,
                      scoring="average_precision", random_state=0,
                      drop_columns=drop_columns)
    model = trainer.train(X_train, y_train)
    features = X_test.iloc[0].to_dict()

    contributions = feature_contributions(model, features)
    score = model.decision_function(X_test.iloc[[0]])[0]
    # The model's score is the intercept plus the sum of the contributions.
    assert np.isclose(contributions.sum() + model["model"].intercept_[0], score)


APP_PATH = str(PROJECT_ROOT / "src" / "spamjam" / "app.py")


def test_page_runs_without_errors():
    page = AppTest.from_file(APP_PATH)
    page.run(timeout=30)
    assert not page.exception


def test_page_predicts_a_pasted_email():
    page = AppTest.from_file(APP_PATH)
    page.run(timeout=30)
    page.radio[0].set_value("Paste an email").run(timeout=30)
    email = (PROJECT_ROOT / "examples" / "new_email.txt").read_text(encoding="utf-8")
    page.text_area[0].input(email)
    check = next(button for button in page.button if button.label == BUTTON_CHECK)
    check.click().run(timeout=30)
    assert not page.exception
    assert any("Spam probability" in bar.text for bar in page.get("progress"))


def test_page_runs_with_the_second_version():
    page = AppTest.from_file(APP_PATH)
    page.run(timeout=30)
    page.toggle[0].set_value(True).run(timeout=30)
    assert not page.exception
    assert any("82.0%" in bar.text for bar in page.get("progress"))   # its threshold
