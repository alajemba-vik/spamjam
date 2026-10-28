"""Tests for the Trainer: build, train, evaluate, save and load."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spamjam.config import load_config
from spamjam.evaluate import compute_metrics
from spamjam.models import Trainer, make_model

# Dynamically load the random_state from config
RANDOM_STATE = load_config()["random_state"]

SMALL_GRID = {"model__C": [0.1, 1.0]}


@pytest.fixture
def trainer():
    return Trainer("logistic_regression", SMALL_GRID, n_splits=3,
                   scoring="average_precision", random_state=0)


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
    assert np.sum(metrics["confusion_matrix"]) == len(y_test)


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
    assert metrics["confusion_matrix"].tolist() == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5


def test_dummy_classifier_training():
    """
    Test that the DummyClassifier can fit and predict on synthetic data.
    """
    # Generate robust synthetic data (100 samples, 5 features, imbalanced)
    # Using weights=[0.6, 0.4] to simulate the Spambase imbalance (Ham > Spam)
    X_dummy, y_dummy = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=2,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=RANDOM_STATE
    )

    # Initialize and train the DummyClassifier
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    model.fit(X_dummy, y_dummy)

    # Make predictions
    predictions = model.predict(X_dummy)

    # Assertions
    assert len(predictions) == 100, "Should predict one label per sample."
    # Since strategy is 'most_frequent', and Ham (0) is the majority,
    # all predictions should be 0
    assert set(predictions) == {0}, \
        "DummyClassifier should only predict the majority class (0)."


def test_pipeline_no_leakage():
    """
    Test that the StandardScaler inside a Pipeline only fits on the training data.
    """
    # Generate synthetic data
    X_dummy, y_dummy = make_classification(
        n_samples=100,
        n_features=5,
        random_state=RANDOM_STATE
    )

    # Create a pipeline with a scaler and the dummy model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',
         DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
    ])

    # 1. Before fitting, the scaler should NOT have calculated any means (no leakage)
    assert not hasattr(pipeline.named_steps['scaler'], 'mean_'), \
        "Scaler should not be fitted before the pipeline is trained."

    # 2. Fit the pipeline
    pipeline.fit(X_dummy, y_dummy)

    # 3. After fitting, the scaler SHOULD have calculated the means
    assert hasattr(pipeline.named_steps['scaler'], 'mean_'), \
        "Scaler should be fitted after the pipeline is trained."


def test_pipeline_handles_invalid_data():
    """
    Test that the pipeline raises an error when given invalid data (e.g., 1D array).
    """
    # Create a 1D array instead of a 2D array (features must be 2D)
    X_invalid = [1.0, 2.0, 3.0]
    y_dummy = [0, 1, 0]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyClassifier(strategy="most_frequent"))
    ])

    # Assert that a ValueError is raised
    with pytest.raises(ValueError):
        pipeline.fit(X_invalid, y_dummy)
