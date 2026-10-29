"""Tests for the Trainer: build, train, evaluate, save and load."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score

from spamjam.constants import SPAM
from spamjam.evaluate import CONFUSION_MATRIX, compute_metrics
from spamjam.models import Trainer, make_model
from spamjam.config import load_config

SMALL_GRID = {"model__C": [0.1, 1.0]}
MODEL_NAMES = ["logistic_regression", "knn", "linear_svm"]
config=load_config()


@pytest.fixture
def trainer():
    config = load_config()
    return Trainer(
        model_name="logistic_regression",
        param_grid=SMALL_GRID,
        n_splits=config["cv"]["n_splits"],
        scoring=config["cv"]["scoring"],
        random_state=config["random_state"]
    )


def test_from_config_reads_the_settings():
    config = load_config()
    trainer = Trainer.from_config()
    assert trainer.model_name == config["model"]["name"]
    assert trainer.param_grid == config["grids"][config["model"]["name"]]
    assert trainer.n_splits == config["cv"]["n_splits"]
    assert trainer.random_state == config["random_state"]


def test_model_without_a_grid_is_rejected():
    with pytest.raises(ValueError, match="No grid"):
        Trainer.from_config(model_name="not_in_config")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_can_be_built_from_config(name):
    assert Trainer.from_config(model_name=name).model_name == name


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_probabilities(name, train_test):
    X_train, y_train, X_test, _ = train_test
    trainer = Trainer(name, {}, n_splits=3, scoring="average_precision",
                      random_state=0)
    proba = trainer.train(X_train, y_train).predict_proba(X_test)

    # One row per email, one column per label (ham, spam).
    assert proba.shape == (len(X_test), 2)
    # Each value is a probability, and each row adds up to 1.
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("not_a_model", random_state=0)


def test_train_picks_a_value_from_the_grid(trainer, train_test):
    X_train, y_train, _, _ = train_test
    model = trainer.train(X_train, y_train)
    assert trainer.search.best_params_["model__C"] in SMALL_GRID["model__C"]
    assert model is trainer.model


def test_evaluate_returns_metrics_in_range(trainer, train_test):
    X_train, y_train, X_test, y_test = train_test
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test, threshold=0.5)
    for name in ["precision", "recall", "f1", "roc_auc", "average_precision"]:
        assert 0 <= metrics[name] <= 1
    assert np.sum(metrics[CONFUSION_MATRIX]) == len(y_test)


def test_evaluate_before_training_is_rejected(trainer, train_test):
    _, _, X_test, y_test = train_test
    with pytest.raises(RuntimeError):
        trainer.evaluate(X_test, y_test, threshold=0.5)


def test_saved_model_gives_the_same_predictions(trainer, train_test, tmp_path):
    X_train, y_train, X_test, _ = train_test
    model = trainer.train(X_train, y_train)
    trainer.save(tmp_path / "model.joblib")

    loaded = Trainer("logistic_regression", {}, 3, "average_precision", 0)
    loaded.load(tmp_path / "model.joblib")
    assert np.allclose(
        model.predict_proba(X_test), loaded.model.predict_proba(X_test)
    )


def test_metrics_follow_the_threshold():
    y_true = [0, 0, 1, 1]
    spam_probability = [0.1, 0.6, 0.4, 0.9]
    metrics = compute_metrics(y_true, spam_probability, threshold=0.5)
    # predicted: 0, 1, 0, 1 -> 1 true ham, 1 false spam, 1 missed spam, 1 true spam
    assert metrics[CONFUSION_MATRIX].tolist() == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5


def test_model_beats_dummy_baseline(trainer, train_test):
    X_train, y_train, X_test, y_test = train_test
    trainer.train(X_train, y_train)
    model_ap = trainer.evaluate(X_test, y_test, threshold=0.5)["average_precision"]

    # The dummy always predicts ham: its average precision is the spam rate.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_ap = average_precision_score(y_test, dummy.predict_proba(X_test)[:, SPAM])
    assert model_ap > dummy_ap

def test_evaluate_with_different_thresholds(trainer, train_test):
    X_train, y_train, X_test, y_test = train_test
    trainer.train(X_train, y_train)

    probas = trainer.model.predict_proba(X_test)[:, SPAM]

    threshold_low = np.percentile(probas, 25)  # Lenient: accepts more spam
    threshold_high = np.percentile(probas, 75) # Strict: only accepts highly probable spam

    #Evaluate with both thresholds
    metrics_low = trainer.evaluate(X_test, y_test, threshold=threshold_low)
    metrics_high = trainer.evaluate(X_test, y_test, threshold=threshold_high)

    # Extract the confusion matrices
    cm_low = metrics_low[CONFUSION_MATRIX]
    cm_high = metrics_high[CONFUSION_MATRIX]

    tp_low, fp_low = cm_low[1, 1], cm_low[0, 1]
    tp_high, fp_high = cm_high[1, 1], cm_high[0, 1]

    assert (tp_high + fp_high) <= (tp_low + fp_low)

    assert fp_high <= fp_low

    assert metrics_high["precision"] >= metrics_low["precision"]

    assert metrics_high["recall"] <= metrics_low["recall"]