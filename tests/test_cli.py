"""Tests for the command-line interface: train, evaluate and predict."""

import json

import pandas as pd
import pytest
import yaml

from spamjam.cli import main, show_metrics
from spamjam.config import load_config
from spamjam.constants import PROJECT_ROOT, TARGET_COL
from spamjam.data import DatasetLoader
from spamjam.evaluate import CONFUSION_MATRIX


def test_train_saves_both_versions(trained):
    config_path, folder = trained
    assert (folder / "model.joblib").exists()
    assert (folder / "model_without_donor_words.joblib").exists()


def test_evaluate_prints_the_test_metrics(trained, capsys):
    config_path, _ = trained
    main(["--config", str(config_path), "evaluate"])
    output = capsys.readouterr().out
    for name in ["precision", "recall", "f1", "roc_auc", "average_precision"]:
        assert name in output
    assert "predicted spam" in output
    assert "without the donor's words" in output


def test_predict_prints_a_probability_and_a_label(trained, capsys):
    config_path, folder = trained
    _, test_df = DatasetLoader.from_config(config_path).load_split()  # temporary split
    email = test_df.drop(columns=TARGET_COL).iloc[0].to_dict()
    input_path = folder / "new_case.json"
    input_path.write_text(json.dumps(email))

    capsys.readouterr()  # ignore what earlier steps printed
    main(["--config", str(config_path), "predict", "--input", str(input_path)])
    result = json.loads(capsys.readouterr().out)
    assert 0 <= result["spam_probability"] <= 1
    assert result["is_spam"] in (0, 1)


def test_predict_reads_the_text_of_an_email(trained, capsys):
    config_path, _ = trained
    text_path = PROJECT_ROOT / "examples" / "new_email.txt"

    capsys.readouterr()  # ignore what earlier steps printed
    main(["--config", str(config_path), "predict", "--text", str(text_path)])
    result = json.loads(capsys.readouterr().out)
    assert 0 <= result["spam_probability"] <= 1
    assert result["is_spam"] in (0, 1)


def test_predict_with_the_second_version(trained, capsys):
    config_path, _ = trained
    text_path = PROJECT_ROOT / "examples" / "new_email.txt"
    capsys.readouterr()
    main(["--config", str(config_path), "predict", "--text", str(text_path),
          "--without-donor-words"])
    result = json.loads(capsys.readouterr().out)
    assert 0 <= result["spam_probability"] <= 1


def test_predict_a_folder_of_emails(trained, tmp_path):
    config_path, _ = trained
    emails = tmp_path / "emails"
    emails.mkdir()
    (emails / "a.txt").write_text("Get FREE money today!!!", encoding="utf-8")
    (emails / "b.txt").write_text("See you at the meeting.", encoding="utf-8")
    output = tmp_path / "predictions.csv"
    main(["--config", str(config_path), "predict", "--text", str(emails),
          "--output", str(output)])
    predictions = pd.read_csv(output)
    assert predictions["email"].tolist() == ["a.txt", "b.txt"]
    assert set(predictions["is_spam"]) <= {0, 1}


def test_features_saves_a_dataset_in_the_spambase_format(trained, tmp_path):
    config_path, _ = trained
    for label in ["spam", "ham"]:
        (tmp_path / "emails" / label).mkdir(parents=True)
        (tmp_path / "emails" / label / "1.txt").write_text("hello", encoding="utf-8")
    output = tmp_path / "dataset.csv"
    main(["--config", str(config_path), "features", str(tmp_path / "emails"),
          "--output", str(output)])
    dataset = pd.read_csv(output)
    real_columns = pd.read_csv(PROJECT_ROOT / load_config()["data"]["path"], nrows=1)
    assert list(dataset.columns) == list(real_columns.columns)
    assert len(dataset) == 2


def test_predict_rejects_a_bad_input_file(trained, tmp_path):
    config_path, _ = trained
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps({"word_freq_free": 1.0}))
    with pytest.raises(SystemExit, match="Missing values"):
        main(["--config", str(config_path), "predict", "--input", str(input_path)])


def test_evaluate_without_a_model_fails(trained, tmp_path):
    # The temporary config, so the real data/processed split is never touched.
    config = load_config(trained[0])
    config["model"]["path"] = str(tmp_path / "no_model.joblib")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit, match="spamjam train"):
        main(["--config", str(config_path), "evaluate"])


def test_missing_config_gives_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["--config", str(tmp_path / "config.yaml"), "train"])


def test_show_metrics_prints_scores_and_a_table(capsys):
    metrics = {"precision": 0.9, "recall": 0.8, CONFUSION_MATRIX: [[50, 5], [10, 35]]}
    show_metrics(metrics)
    output = capsys.readouterr().out
    assert "precision" in output and "0.9" in output
    assert "predicted ham  predicted spam" in output
    assert "actual spam" in output

