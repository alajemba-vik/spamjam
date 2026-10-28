import pytest
import yaml
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from spamjam import constants

# Dynamically load the random_state from config
with open(constants.CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    RANDOM_STATE = config["random_state"]


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
    # Since strategy is 'most_frequent', and Ham (0) is the majority, all predictions should be 0
    assert set(predictions) == {0}, "DummyClassifier should only predict the majority class (0)."


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
        ('model', DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))
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