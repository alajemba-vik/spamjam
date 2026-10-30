"""Command-line interface for SpamJam.

    spamjam train                     train the model and save it
    spamjam evaluate                  print the metrics on the test set
    spamjam predict --input FILE      predict one new email from a JSON file
    spamjam predict --text FILE       predict one new email from its text

Every setting (data paths, model, grid, threshold) comes from config.yaml.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from spamjam.config import load_config
from spamjam.constants import CONFIG_PATH, PROJECT_ROOT, TARGET_COL
from spamjam.data import DatasetLoader
from spamjam.evaluate import CONFUSION_MATRIX
from spamjam.models import Trainer
from spamjam.predict import predict_email, read_email_features
from spamjam.text import features_from_text

# Commands, as typed after `spamjam`.
CMD_TRAIN = "train"
CMD_EVALUATE = "evaluate"
CMD_PREDICT = "predict"

# Help texts shown by `spamjam --help`.
HELP_DESCRIPTION = "Spam detection for GrandPied."
HELP_CONFIG = "settings file (default: the project's config.yaml)"
HELP_TRAIN = "train the model and save it"
HELP_EVALUATE = "print the metrics on the test set"
HELP_PREDICT = "predict one new email"
HELP_INPUT = (
    "JSON file with the email's 57 feature values, like examples/new_case.json "
    "(every feature is described in docs/data_dictionary.md)"
)
HELP_TEXT = (
    "plain text file with the email itself, like examples/new_email.txt; "
    "SpamJam computes the 57 features from it"
)

# Messages printed by the commands.
MSG_CV_SCORE = "CV {scoring}: {score:.4f}"
MSG_MODEL_SAVED = "Model saved to {path}"
MSG_TEST_SET = "Test set: {n_emails} emails, threshold {threshold}"

# Labels of the confusion matrix table: rows are the true labels, columns the
# predictions, both in the order HAM, SPAM (as in evaluate.py).
LABELS_CONFUSION_ROWS = ["actual ham", "actual spam"]
LABELS_CONFUSION_COLUMNS = ["predicted ham", "predicted spam"]

# Errors: spamjam stops with one of these instead of a traceback.
ERROR_NO_CONFIG = (
    "config.yaml not found. Run spamjam from the project folder, "
    "or pass --config path/to/config.yaml."
)
ERROR_NO_MODEL = "No model at {path}. Run `spamjam train` first."
ERROR_CANNOT_PREDICT = "Cannot predict: {error}"

# Decimals shown for scores and probabilities.
DECIMALS = 4


# --- Helpers -----------------------------------------------------------------


def load_trained_model(trainer: Trainer, model_path: Path) -> None:
    """Load the saved model into the trainer, or stop if there is none."""
    if not model_path.exists():
        sys.exit(ERROR_NO_MODEL.format(path=model_path))
    trainer.load(model_path)


def show_metrics(metrics: dict) -> None:
    """Print each score, then the confusion matrix as a table."""
    scores = pd.Series(
        {name: value for name, value in metrics.items() if name != CONFUSION_MATRIX}
    )
    print(scores.round(DECIMALS).to_string())
    print()

    confusion = pd.DataFrame(
        metrics[CONFUSION_MATRIX],
        index=LABELS_CONFUSION_ROWS,
        columns=LABELS_CONFUSION_COLUMNS,
    )
    print(confusion.to_string())


# --- Commands ----------------------------------------------------------------


def train(trainer: Trainer, loader: DatasetLoader, model_path: Path) -> None:
    """Train the model on the training set and save it."""
    train_df, _ = loader.load_or_create_split()
    trainer.train(train_df.drop(columns=TARGET_COL), train_df[TARGET_COL])
    trainer.save(model_path)
    score = trainer.search.best_score_
    print(MSG_CV_SCORE.format(scoring=trainer.scoring, score=score))
    print(MSG_MODEL_SAVED.format(path=model_path))


def evaluate(
    trainer: Trainer, loader: DatasetLoader, model_path: Path, threshold: float
) -> None:
    """Print the saved model's metrics on the test set."""
    load_trained_model(trainer, model_path)
    _, test_df = loader.load_or_create_split()
    metrics = trainer.evaluate(
        test_df.drop(columns=TARGET_COL), test_df[TARGET_COL], threshold
    )
    print(MSG_TEST_SET.format(n_emails=len(test_df), threshold=threshold))
    print()
    show_metrics(metrics)


def predict(
    trainer: Trainer,
    model_path: Path,
    threshold: float,
    input_path: Path | None = None,
    text_path: Path | None = None,
) -> None:
    """Print the prediction for one email, from a JSON file or a text file."""
    load_trained_model(trainer, model_path)
    try:
        if text_path is not None:
            text = text_path.read_text(encoding="utf-8")
            feature_names = list(trainer.model.feature_names_in_)
            features = features_from_text(text, feature_names)
        else:
            features = read_email_features(input_path)
        result = predict_email(trainer.model, features, threshold)
    except (OSError, ValueError) as error:
        sys.exit(ERROR_CANNOT_PREDICT.format(error=error))
    result["spam_probability"] = round(result["spam_probability"], DECIMALS)
    print(json.dumps(result))


# --- Entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Describe the commands and options that `spamjam` accepts."""
    parser = argparse.ArgumentParser(prog="spamjam", description=HELP_DESCRIPTION)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help=HELP_CONFIG)

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(CMD_TRAIN, help=HELP_TRAIN)
    commands.add_parser(CMD_EVALUATE, help=HELP_EVALUATE)
    predict_parser = commands.add_parser(CMD_PREDICT, help=HELP_PREDICT)
    # One of the two, not both.
    email_input = predict_parser.add_mutually_exclusive_group(required=True)
    email_input.add_argument("--input", type=Path, metavar="FILE", help=HELP_INPUT)
    email_input.add_argument("--text", type=Path, metavar="FILE", help=HELP_TEXT)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Read the command line and the settings, then run the chosen command.

    ``argv`` lets the tests pass arguments directly; by default argparse
    reads them from the command line.
    """
    args = build_parser().parse_args(argv)
    if not args.config.exists():
        sys.exit(ERROR_NO_CONFIG)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(args.config)
    model_path = PROJECT_ROOT / config["model"]["path"]
    threshold = config["model"]["threshold"]
    loader = DatasetLoader.from_config(args.config)
    trainer = Trainer.from_config(config_path=args.config)

    if args.command == CMD_TRAIN:
        train(trainer, loader, model_path)
    elif args.command == CMD_EVALUATE:
        evaluate(trainer, loader, model_path, threshold)
    elif args.command == CMD_PREDICT:
        predict(trainer, model_path, threshold, args.input, args.text)


if __name__ == "__main__":
    main()
