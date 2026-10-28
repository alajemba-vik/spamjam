"""Train, evaluate, save and load SpamJam models."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from spamjam.config import load_config
from spamjam.constants import CONFIG_PATH, SPAM
from spamjam.evaluate import compute_metrics
from spamjam.features import build_pipeline

logger = logging.getLogger(__name__)


def make_model(name: str, random_state: int) -> BaseEstimator:
    """Return a new, untrained model from its name in config.yaml."""
    if name == "logistic_regression":
        # liblinear supports both L1 and L2, which our grid compares.
        return LogisticRegression(solver="liblinear", random_state=random_state)
    raise ValueError(f"Unknown model {name!r}.")


class Trainer:
    """Build the pipeline for one model, tune it, evaluate it, save and load it.

    Training runs a grid search with stratified cross-validation on the
    training set only, then keeps the best pipeline refitted on all of it.
    """

    def __init__(
        self,
        model_name: str,
        param_grid: dict,
        n_splits: int,
        scoring: str,
        random_state: int,
    ) -> None:
        self.model_name = model_name
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        self.search = None  # the GridSearchCV, after train()
        self.model = None  # the best pipeline, after train() or load()

    @classmethod
    def from_config(
        cls, model_name: str | None = None, config_path: Path = CONFIG_PATH
    ) -> "Trainer":
        """Create a Trainer from config.yaml, by default for its ``model.name``."""
        config = load_config(config_path)
        name = model_name or config["model"]["name"]
        if name not in config["grids"]:
            raise ValueError(f"No grid for {name!r} in config.yaml.")
        return cls(
            model_name=name,
            param_grid=config["grids"][name],
            n_splits=config["cv"]["n_splits"],
            scoring=config["cv"]["scoring"],
            random_state=config["random_state"],
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Tune the model with cross-validation and return the best pipeline."""
        pipeline = build_pipeline(make_model(self.model_name, self.random_state))
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        self.search = GridSearchCV(
            pipeline, self.param_grid, cv=cv, scoring=self.scoring
        )
        self.search.fit(X, y)
        self.model = self.search.best_estimator_
        logger.info("Best parameters: %s", self.search.best_params_)
        return self.model

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
        """Return the metrics of the trained (or loaded) model on X and y."""
        if self.model is None:
            raise RuntimeError("Train or load a model before evaluating it.")
        # predict_proba has one column per label (HAM, SPAM); keep the spam one.
        spam_probability = self.model.predict_proba(X)[:, SPAM]
        return compute_metrics(y, spam_probability, threshold)

    def save(self, path: Path) -> None:
        """Save the trained model to ``path`` with joblib."""
        if self.model is None:
            raise RuntimeError("Train a model before saving it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> Pipeline:
        """Load a saved model from ``path``, keep it and return it."""
        self.model = joblib.load(path)
        return self.model
