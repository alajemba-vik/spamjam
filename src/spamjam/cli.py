"""Command-line interface for SpamJam.

    spamjam train                       train both versions of the model and save them
    spamjam evaluate                    print both versions' metrics on the test set
    spamjam predict --input FILE        predict one new email from a JSON file
    spamjam predict --text FILE         predict one new email from its text
    spamjam predict --text EMAILS       predict every email in a folder or .mbox file
    spamjam features EMAILS --output F  save the features of many emails as a dataset

The two versions are the model with all 57 features and the same model
without the donor's words (see docs/limitations.md). ``predict`` uses the
first one, or the second with ``--without-donor-words``.

Every setting (data paths, model, grid, thresholds) comes from config.yaml.
"""

import argparse
import json
import logging
import sys
from email import message_from_bytes, policy
from pathlib import Path

import pandas as pd

from spamjam.config import load_config
from spamjam.constants import CONFIG_PATH, SPAM, TARGET_COL
from spamjam.data import DatasetLoader
from spamjam.emails import (
    EMAIL_COL,
    EML_FILE,
    MBOX_FILE,
    features_table,
    message_text,
    read_emails,
)
from spamjam.evaluate import CONFUSION_MATRIX
from spamjam.models import Trainer, version_settings
from spamjam.predict import SPAM_PROBABILITY, predict_email, read_email_features
from spamjam.text import features_from_text

# Commands, as typed after `spamjam`.
CMD_TRAIN = "train"
CMD_EVALUATE = "evaluate"
CMD_PREDICT = "predict"
CMD_FEATURES = "features"

# The two versions of the model: (without_donor_words, name as printed).
VERSIONS = [
    (False, "Model with all features"),
    (True, "Model without the donor's words"),
]

# Help texts shown by `spamjam --help`.
HELP_DESCRIPTION = "Spam detection for GrandPied."
HELP_CONFIG = "settings file (default: the project's config.yaml)"
HELP_TRAIN = "train both versions of the model and save them"
HELP_EVALUATE = "print both versions' metrics on the test set"
HELP_PREDICT = "predict one new email, or every email in a folder or .mbox file"
HELP_FEATURES = "compute the 57 features of many emails and save them as a dataset"
HELP_INPUT = (
    "JSON file with the email's 57 feature values, like examples/new_case.json "
    "(every feature is described in docs/data_dictionary.md)"
)
HELP_TEXT = (
    "the email itself (a .txt or .eml file, like examples/new_email.txt), or a "
    "folder of them, or an .mbox file; SpamJam computes the 57 features"
)
HELP_NO_DONOR = "use the version trained without the donor's words"
HELP_EMAILS = (
    "folder of .txt/.eml emails, or an .mbox file; emails in a spam/ or ham/ "
    "folder (or spam.mbox, ham.mbox) get that label"
)
HELP_OUTPUT_PREDICT = "save the predictions of many emails to this CSV file"
HELP_OUTPUT_FEATURES = "CSV file to save the features to"

# Messages printed by the commands.
MSG_VERSION = "== {version} =="
MSG_CV_SCORE = "CV {scoring}: {score:.4f}"
MSG_MODEL_SAVED = "Model saved to {path}"
MSG_TEST_SET = "Test set: {n_emails} emails, threshold {threshold}"
MSG_SAVED = "Saved {n_emails} emails to {path}"
MSG_NO_LABELS = (
    "Not every email is in a spam/ or ham/ folder, so the file has no is_spam column."
)

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
ERROR_CANNOT_READ = "Cannot read the emails: {error}"

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


def read_email_text(path: Path) -> str:
    """Return the text of one email file (.eml files without their headers)."""
    if path.suffix == EML_FILE:
        message = message_from_bytes(path.read_bytes(), policy=policy.default)
        return message_text(message)
    return path.read_text(encoding="utf-8")


def feature_names(config_path: Path) -> list[str]:
    """Return the 57 Spambase feature names, in the dataset's order."""
    columns = DatasetLoader.from_config(config_path).load_data().columns
    return [name for name in columns if name != TARGET_COL]


# --- Commands ----------------------------------------------------------------


def train(config_path: Path) -> None:
    """Train both versions of the model on the training set and save them."""
    train_df, _ = DatasetLoader.from_config(config_path).load_or_create_split()
    config = load_config(config_path)
    for without_donor_words, version in VERSIONS:
        trainer = Trainer.from_config(
            config_path=config_path, without_donor_words=without_donor_words
        )
        trainer.train(train_df.drop(columns=TARGET_COL), train_df[TARGET_COL])
        model_path = version_settings(config, without_donor_words)["path"]
        trainer.save(model_path)
        print(MSG_VERSION.format(version=version))
        score = trainer.search.best_score_
        print(MSG_CV_SCORE.format(scoring=trainer.scoring, score=score))
        print(MSG_MODEL_SAVED.format(path=model_path))
        print()


def evaluate(config_path: Path) -> None:
    """Print both versions' metrics on the test set."""
    _, test_df = DatasetLoader.from_config(config_path).load_or_create_split()
    config = load_config(config_path)
    for without_donor_words, version in VERSIONS:
        settings = version_settings(config, without_donor_words)
        trainer = Trainer.from_config(config_path=config_path)
        load_trained_model(trainer, settings["path"])
        metrics = trainer.evaluate(
            test_df.drop(columns=TARGET_COL), test_df[TARGET_COL], settings["threshold"]
        )
        print(MSG_VERSION.format(version=version))
        threshold = settings["threshold"]
        print(MSG_TEST_SET.format(n_emails=len(test_df), threshold=threshold))
        print()
        show_metrics(metrics)
        print()


def predict(args: argparse.Namespace, config_path: Path) -> None:
    """Print the prediction for one email, or the predictions for many."""
    settings = version_settings(load_config(config_path), args.without_donor_words)
    trainer = Trainer.from_config(config_path=config_path)
    load_trained_model(trainer, settings["path"])
    names = list(trainer.model.feature_names_in_)

    if args.text is not None and (args.text.is_dir() or args.text.suffix == MBOX_FILE):
        try:
            emails = read_emails(args.text)
        except (OSError, ValueError) as error:
            sys.exit(ERROR_CANNOT_READ.format(error=error))
        table = features_table(emails, names)[names]
        probabilities = trainer.model.predict_proba(table)
        results = pd.DataFrame({
            EMAIL_COL: emails[EMAIL_COL],
            SPAM_PROBABILITY: probabilities[:, SPAM].round(DECIMALS),
            TARGET_COL: (probabilities[:, SPAM] >= settings["threshold"]).astype(int),
        })
        if args.output:
            results.to_csv(args.output, index=False)
            print(MSG_SAVED.format(n_emails=len(results), path=args.output))
        else:
            print(results.to_string(index=False))
        return

    try:
        if args.text is not None:
            features = features_from_text(read_email_text(args.text), names)
        else:
            features = read_email_features(args.input)
        result = predict_email(trainer.model, features, settings["threshold"])
    except (OSError, ValueError) as error:
        sys.exit(ERROR_CANNOT_PREDICT.format(error=error))
    result[SPAM_PROBABILITY] = round(result[SPAM_PROBABILITY], DECIMALS)
    print(json.dumps(result))


def features(args: argparse.Namespace, config_path: Path) -> None:
    """Save the features of many emails as a dataset in the Spambase format."""
    try:
        emails = read_emails(args.emails)
    except (OSError, ValueError) as error:
        sys.exit(ERROR_CANNOT_READ.format(error=error))
    table = features_table(emails, feature_names(config_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(MSG_SAVED.format(n_emails=len(table), path=args.output))
    if TARGET_COL not in table:
        print(MSG_NO_LABELS)


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
    email_input.add_argument("--text", type=Path, metavar="EMAILS", help=HELP_TEXT)
    predict_parser.add_argument(
        "--without-donor-words", action="store_true", help=HELP_NO_DONOR
    )
    predict_parser.add_argument(
        "--output", type=Path, metavar="FILE", help=HELP_OUTPUT_PREDICT
    )

    features_parser = commands.add_parser(CMD_FEATURES, help=HELP_FEATURES)
    features_parser.add_argument(
        "emails", type=Path, metavar="EMAILS", help=HELP_EMAILS
    )
    features_parser.add_argument(
        "--output", type=Path, metavar="FILE", required=True, help=HELP_OUTPUT_FEATURES
    )
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

    if args.command == CMD_TRAIN:
        train(args.config)
    elif args.command == CMD_EVALUATE:
        evaluate(args.config)
    elif args.command == CMD_PREDICT:
        predict(args, args.config)
    elif args.command == CMD_FEATURES:
        features(args, args.config)


if __name__ == "__main__":
    main()
