"""Tests for the Streamlit page: explanations, and that it runs."""

import numpy as np
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from spamjam import constants
from spamjam.app import (
    BUTTON_CHECK,
    CHOICE_TEXT,
    ERROR_NO_MODEL,
    feature_contributions,
)
from spamjam.config import load_config
from spamjam.constants import PROJECT_ROOT
from spamjam.evaluate import AVERAGE_PRECISION
from spamjam.models import LOGISTIC_REGRESSION, Trainer, version_settings

APP_PATH = PROJECT_ROOT / "src" / "spamjam" / "app.py"
EXAMPLE_TEXT = PROJECT_ROOT / load_config()["examples"]["text"]
DONOR_COLUMNS = load_config()["features"]["donor_columns"]
TIMEOUT = 30   # seconds the page may take to run


@pytest.mark.parametrize("drop_columns", [None, DONOR_COLUMNS])
def test_contributions_add_up_to_the_model_score(train_test, drop_columns):
    X_train, y_train, X_test, _ = train_test
    trainer = Trainer(LOGISTIC_REGRESSION, {}, n_splits=3,
                      scoring=AVERAGE_PRECISION, random_state=0,
                      drop_columns=drop_columns)
    model = trainer.train(X_train, y_train)
    features = X_test.iloc[0].to_dict()

    contributions = feature_contributions(model, features)
    score = model.decision_function(X_test.iloc[[0]])[0]
    # The model's score is the intercept plus the sum of the contributions.
    assert np.isclose(contributions.sum() + model["model"].intercept_[0], score)


def open_page(config_path, monkeypatch) -> AppTest:
    """Run the page with the given settings file and return it.

    The page reads spamjam.constants.CONFIG_PATH each time it runs, so pointing
    it to a temporary config makes it use that config's models, not models/.
    """
    monkeypatch.setattr(constants, "CONFIG_PATH", config_path)
    page = AppTest.from_file(str(APP_PATH))
    page.run(timeout=TIMEOUT)
    return page


@pytest.fixture
def page(trained, monkeypatch):
    """The page, using the small models trained for the tests."""
    return open_page(trained[0], monkeypatch)


def test_page_shows_a_verdict(page):
    assert not page.exception
    assert len(page.get("progress")) == 1   # the spam probability bar


def test_page_without_a_model_says_to_train(trained, tmp_path, monkeypatch):
    config = load_config(trained[0])
    missing_model = tmp_path / "no_model.joblib"
    config["model"]["path"] = str(missing_model)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    page = open_page(config_path, monkeypatch)
    assert not page.exception
    assert page.error[0].value == ERROR_NO_MODEL.format(path=missing_model)


def test_page_predicts_a_pasted_email(page):
    page.radio[0].set_value(CHOICE_TEXT).run(timeout=TIMEOUT)
    page.text_area[0].input(EXAMPLE_TEXT.read_text(encoding="utf-8"))
    check = next(button for button in page.button if button.label == BUTTON_CHECK)
    check.click().run(timeout=TIMEOUT)
    assert not page.exception
    assert len(page.get("progress")) == 1


def test_page_uses_the_second_versions_threshold(trained, page):
    page.toggle[0].set_value(True).run(timeout=TIMEOUT)
    threshold = version_settings(load_config(trained[0]), True)["threshold"]
    assert not page.exception
    assert f"{threshold:.1%}" in page.get("progress")[0].text
