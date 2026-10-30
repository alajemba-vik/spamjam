"""Train, evaluate, save and load SpamJam models."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from spamjam.config import load_config
from spamjam.constants import CONFIG_PATH, PROJECT_ROOT, SPAM
from spamjam.evaluate import compute_metrics
from spamjam.features import build_pipeline

logger = logging.getLogger(__name__)

# The models we compare, as named in config.yaml (grids, model.name).
LOGISTIC_REGRESSION = "logistic_regression"
KNN = "knn"
LINEAR_SVM = "linear_svm"
MODEL_NAMES = [LOGISTIC_REGRESSION, KNN, LINEAR_SVM]

MSG_BEST_PARAMETERS = "Best parameters: %s"
ERROR_UNKNOWN_MODEL = "Unknown model {name!r}."
ERROR_NO_GRID = "No grid for {name!r} in config.yaml."
ERROR_NOT_TRAINED = "Train or load a model before evaluating it."
ERROR_NOTHING_TO_SAVE = "Train a model before saving it."


def make_model(name: str, random_state: int) -> BaseEstimator:
    """Return a new, untrained model from its name in config.yaml."""
    if name == LOGISTIC_REGRESSION:
        # liblinear supports both L1 and L2, which our grid compares.
        return LogisticRegression(solver="liblinear", random_state=random_state)
    if name == KNN:
        return KNeighborsClassifier()
    if name == LINEAR_SVM:
        # SVC with a linear kernel rather than LinearSVC, because LinearSVC
        # can't give probabilities. probability=True adds them; our metrics
        # and predictions need the probability of spam.
        return SVC(kernel="linear", probability=True, random_state=random_state)
    raise ValueError(ERROR_UNKNOWN_MODEL.format(name=name))


def version_settings(config: dict, without_donor_words: bool = False) -> dict:
    """Return where one version of the model is saved and its threshold.

    The project trains two versions of the same model: one with all 57
    features, and one without the donor's words (``features.donor_columns``),
    see docs/limitations.md.
    """
    settings = config["model"]
    if without_donor_words:
        settings = settings["without_donor_words"]
    return {
        "path": PROJECT_ROOT / settings["path"],
        "threshold": settings["threshold"],
    }


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
        drop_columns: list[str] | None = None,
    ) -> None:
        """Store the settings of the training run.

        Args:
            model_name: model to build, as named in config.yaml.
            param_grid: values to try for each hyperparameter, with pipeline
                names (e.g. ``model__C``). An empty grid keeps the defaults.
            n_splits: number of cross-validation folds.
            scoring: metric used to compare the settings.
            random_state: seed for the folds and the model.
            drop_columns: columns the pipeline removes before training
                (e.g. the donor's words). None keeps all the features.
        """
        self.model_name = model_name
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        self.drop_columns = drop_columns or []
        self.search = None  # the GridSearchCV, after train()
        self.model = None  # the best pipeline, after train() or load()

    @classmethod
    def from_config(
        cls,
        model_name: str | None = None,
        config_path: Path = CONFIG_PATH,
        without_donor_words: bool = False,
    ) -> "Trainer":
        """Create a Trainer from config.yaml, by default for its ``model.name``.

        With ``without_donor_words``, the model is trained without the donor's
        words listed in ``features.donor_columns``.
        """
        config = load_config(config_path)
        name = model_name or config["model"]["name"]
        if name not in config["grids"]:
            raise ValueError(ERROR_NO_GRID.format(name=name))
        return cls(
            model_name=name,
            param_grid=config["grids"][name],
            n_splits=config["cv"]["n_splits"],
            scoring=config["cv"]["scoring"],
            random_state=config["random_state"],
            drop_columns=(
                config["features"]["donor_columns"] if without_donor_words else []
            ),
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Tune the model with cross-validation and return the best pipeline.

        With an empty grid, this just scores the model's default settings.
        """
        pipeline = build_pipeline(
            make_model(self.model_name, self.random_state), self.drop_columns
        )
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        self.search = GridSearchCV(
            pipeline, self.param_grid, cv=cv, scoring=self.scoring
        )
        self.search.fit(X, y)
        self.model = self.search.best_estimator_
        logger.info(MSG_BEST_PARAMETERS, self.search.best_params_)
        return self.model

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
        """Return the metrics of the trained (or loaded) model on X and y."""
        if self.model is None:
            raise RuntimeError(ERROR_NOT_TRAINED)
        # predict_proba has one column per label (HAM, SPAM); keep the spam one.
        spam_probability = self.model.predict_proba(X)[:, SPAM]
        return compute_metrics(y, spam_probability, threshold)

    def save(self, path: Path) -> None:
        """Save the trained model to ``path`` with joblib."""
        if self.model is None:
            raise RuntimeError(ERROR_NOTHING_TO_SAVE)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> Pipeline:
        """Load a saved model from ``path``, keep it and return it."""
        self.model = joblib.load(path)
        return self.model
