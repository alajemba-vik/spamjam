"""Tests for the command-line interface: train, evaluate, predict, features."""

import json
import re

import pandas as pd
import pytest
import yaml

from spamjam.cli import (
    ERROR_NO_CONFIG,
    ERROR_NO_MODEL,
    LABELS_CONFUSION_COLUMNS,
    LABELS_CONFUSION_ROWS,
    VERSIONS,
    main,
    show_metrics,
)
from spamjam.config import load_config
from spamjam.constants import HAM, PROJECT_ROOT, SPAM, TARGET_COL
from spamjam.data import DatasetLoader
from spamjam.emails import EMAIL_COL, LABELS
from spamjam.evaluate import CONFUSION_MATRIX, PRECISION, RECALL, SCORES
from spamjam.models import version_settings
from spamjam.predict import ERROR_MISSING, SPAM_PROBABILITY

EXAMPLE_TEXT = PROJECT_ROOT / load_config()["examples"]["text"]


def run(config_path, *args: str) -> None:
    """Run `spamjam --config CONFIG_PATH ARGS`."""
    main(["--config", str(config_path), *args])


def test_train_saves_both_versions(trained):
    config = load_config(trained[0])
    for without_donor_words, _ in VERSIONS:
        assert version_settings(config, without_donor_words)["path"].exists()


def test_evaluate_prints_the_test_metrics(trained, capsys):
    run(trained[0], "evaluate")
    output = capsys.readouterr().out
    for name in SCORES + LABELS_CONFUSION_COLUMNS:
        assert name in output
    for _, version in VERSIONS:
        assert version in output


def test_predict_prints_a_probability_and_a_label(trained, capsys):
    config_path, folder = trained
    _, test_df = DatasetLoader.from_config(config_path).load_split()  # temporary split
    email = test_df.drop(columns=TARGET_COL).iloc[0].to_dict()
    input_path = folder / "new_case.json"
    input_path.write_text(json.dumps(email))

    capsys.readouterr()  # ignore what earlier steps printed
    run(config_path, "predict", "--input", str(input_path))
    result = json.loads(capsys.readouterr().out)
    assert 0 <= result[SPAM_PROBABILITY] <= 1
    assert result[TARGET_COL] in (HAM, SPAM)


@pytest.mark.parametrize("version_option", [[], ["--without-donor-words"]])
def test_predict_reads_the_text_of_an_email(trained, capsys, version_option):
    capsys.readouterr()
    run(trained[0], "predict", "--text", str(EXAMPLE_TEXT), *version_option)
    result = json.loads(capsys.readouterr().out)
    assert 0 <= result[SPAM_PROBABILITY] <= 1
    assert result[TARGET_COL] in (HAM, SPAM)


def test_predict_a_folder_of_emails(trained, tmp_path):
    emails = tmp_path / "emails"
    emails.mkdir()
    (emails / "a.txt").write_text("Get FREE money today!!!", encoding="utf-8")
    (emails / "b.txt").write_text("See you at the meeting.", encoding="utf-8")
    output = tmp_path / "predictions.csv"
    run(trained[0], "predict", "--text", str(emails), "--output", str(output))
    predictions = pd.read_csv(output)
    assert predictions[EMAIL_COL].tolist() == ["a.txt", "b.txt"]
    assert set(predictions[TARGET_COL]) <= {HAM, SPAM}


def test_features_saves_a_dataset_in_the_spambase_format(trained, tmp_path, real_df):
    for label in LABELS:   # one email in spam/, one in ham/
        (tmp_path / "emails" / label).mkdir(parents=True)
        (tmp_path / "emails" / label / "1.txt").write_text("hello", encoding="utf-8")
    output = tmp_path / "dataset.csv"
    run(trained[0], "features", str(tmp_path / "emails"), "--output", str(output))
    dataset = pd.read_csv(output)
    assert list(dataset.columns) == list(real_df.columns)
    assert len(dataset) == len(LABELS)


def test_predict_rejects_a_bad_input_file(trained, tmp_path):
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps({"word_freq_free": 1.0}))
    # format(names="") leaves the start of the message, before the list of names.
    with pytest.raises(SystemExit, match=re.escape(ERROR_MISSING.format(names=""))):
        run(trained[0], "predict", "--input", str(input_path))


def test_evaluate_without_a_model_fails(trained, tmp_path):
    # The temporary config, so the real data/processed split is never touched.
    config = load_config(trained[0])
    missing_model = tmp_path / "no_model.joblib"
    config["model"]["path"] = str(missing_model)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    error = ERROR_NO_MODEL.format(path=missing_model)
    with pytest.raises(SystemExit, match=re.escape(error)):
        run(config_path, "evaluate")


def test_missing_config_gives_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match=re.escape(ERROR_NO_CONFIG)):
        run(tmp_path / "config.yaml", "train")


def test_show_metrics_prints_scores_and_a_table(capsys):
    metrics = {PRECISION: 0.9, RECALL: 0.8, CONFUSION_MATRIX: [[50, 5], [10, 35]]}
    show_metrics(metrics)
    output = capsys.readouterr().out
    for name in [PRECISION, RECALL] + LABELS_CONFUSION_ROWS + LABELS_CONFUSION_COLUMNS:
        assert name in output
