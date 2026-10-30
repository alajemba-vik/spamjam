# spamjam
SpamJam is a machine learning project for email spam detection, featuring reproducible data pipelines, model comparison, cost-aware classification, automated testing, and a Python command-line interface. To enable a more immersive learning experience, students have framed the problem around a fictional company GrandPied.


# Problem
GrandPied is a marketing company that relies heavily on email for its daily operations. Recently, its employees have been getting more emails including unsolicited ads, spam and harmful messages.
However, blocking legitimate emails can also be an issue since this could result in a missed client inquiry, or even a lost business opportunity.
To address this challenge, GrandPied's IT department is exploring SpamJam, an ML-based spam detection system.
The goal is to achieve an appropriate balance between identifying unwanted messages and protecting important business communications.

**Scope:** SpamJam is a proof of concept. We build and evaluate the full pipeline on the public Spambase dataset, whose legitimate emails all come from one person (George, at HP Labs, around 1999). Before real use, GrandPied would retrain the model on its own labelled emails with the same code.
- **Target**: Whether an email is spam or ham (legitimate), 1 and 0 respectively.
- **Task**: Supervised binary classification where we help IT admins identify unwanted emails while protecting legitimate business communication
 - **Cost**: GrandPied considers false positives (legitimate emails incorrectly classified as spam) more costly than false negatives (spam incorrectly classified as legitimate email) because it relies heavily on receiving client emails. Although false negatives allow unwanted messages through, GrandPied accepts this trade-off.


# Data
We use the [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) from the UCI Machine Learning Repository (Hopkins, Reeber, Forman and Suermondt, 1999), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It contains 4,601 emails described by 57 numeric features (word and character frequencies, and runs of capital letters).

- 391 exact duplicates are removed, leaving 4,210 emails (about 40 % spam).
- They are split once into a training set (3,368 emails) and a test set (842 emails), stratified so both keep the same share of spam, with the seed from `config.yaml`. The split is saved in `data/processed/`, so the notebooks and the command line all use the same one.
- Every column is described in the data dictionary: [`docs/data_dictionary.md`](docs/data_dictionary.md).

**Limitation:** some features describe the dataset's donor rather than spam (e.g. `george`, `hp`, `650`). They help on Spambase but would not on GrandPied's emails, so our scores are likely optimistic for GrandPied. See `notebooks/analysis.ipynb` for the comparison with and without them.


# Results

| | |
|---|---|
| Final model | Logistic regression (L1, `C`=1), with a log transform of the capital-letter counts and standard scaling |
| Model selection | Average precision 0.954 in 5-fold cross-validation on the training set (dummy baseline: 0.399). Tuned k-NN and linear SVM are tied with it; we keep the simplest |
| Decision threshold | 0.782, chosen on out-of-fold predictions to reach a precision of 0.95 on spam |
| Test set (used once) | Precision 0.950, recall 0.795: 14 of 506 legitimate emails flagged as spam, 69 of 336 spam missed |

The full analysis is in `notebooks/analysis.ipynb`. These scores are likely optimistic for GrandPied (see the limitation above).


# Installation

Requires Python 3.10 to 3.14.

```bash
git clone https://github.com/alajemba-vik/spamjam.git
cd spamjam
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # the package, its dependencies and dev tools (ruff, Jupyter kernel)
```

`pip install -e .` installs the package and the `spamjam` command. The `[dev]` extra adds ruff and the Jupyter kernel used by the notebooks. All versions are pinned in `pyproject.toml`.

With [uv](https://docs.astral.sh/uv/), `uv run spamjam ...` and `uv run pytest` work directly, without the steps above.


# Usage

Run the commands from the project folder (the one that contains `config.yaml`). Every setting (data paths, model, grid, threshold) comes from `config.yaml`; `--config other.yaml` uses another settings file.

**Train** the model on the training set and save it to `models/spamjam.joblib`. On the first run, it also downloads the data if needed and creates the train/test split.

```bash
spamjam train
```

**Evaluate** the saved model on the test set (scores and confusion matrix):

```bash
spamjam evaluate
```

```
Test set: 842 emails, threshold 0.782

precision            0.9502
recall               0.7946
f1                   0.8655
roc_auc              0.9765
average_precision    0.9585

             predicted ham  predicted spam
actual ham             492              14
actual spam             69             267
```

**Predict** one new email from a JSON file with its 57 feature values (see `examples/new_case.json`):

```bash
spamjam predict --input examples/new_case.json
```

```
{"spam_probability": 0.9777, "is_spam": 1}
```

The model never sees the email's text, only its 57 features, so the input is those values, each with its feature name. A missing, unknown or non-numeric value is rejected with a clear message instead of being guessed.


# Tests and code quality

```bash
pytest        # 40 tests: data, pipeline (no leakage), training, metrics, threshold, prediction, command line
ruff check .  # style (PEP 8), errors, import order, names and docstrings, notebooks included
```

The tests that use the command line work on a small temporary split, so they never touch the real test set or your saved model.


# Notebooks

1. `notebooks/data_exploration.ipynb`: the data, its quality, duplicates, skewed features, donor words, and the train/test split.
2. `notebooks/analysis.ipynb`: model selection with cross-validation, the decision threshold, the one-time test set evaluation, and the interpretation.

Open them with the Jupyter kernel of the virtual environment. In `analysis.ipynb`, the cell that tunes all models takes a few minutes (mostly the linear SVM); it prints its progress.


# Project structure

```
config.yaml              all settings: paths, seed, CV, grids, final model, threshold
src/spamjam/
    data.py              DatasetLoader: download, remove duplicates, split
    features.py          preprocessing pipeline (log transform, scaling)
    models.py            Trainer: tune with cross-validation, evaluate, save, load
    evaluate.py          metrics for a given threshold
    predict.py           prediction for one email read from JSON
    cli.py               the `spamjam` command
    config.py            reads config.yaml
    constants.py         project paths and labels
tests/                   pytest tests
notebooks/               data exploration and analysis
examples/new_case.json   example input for `spamjam predict`
docs/data_dictionary.md  what every column of the dataset means
```


# Work Breakdown

https://github.com/users/alajemba-vik/projects/5/views/1


# Quick Checklist
- [ ] A clean pipeline
- [ ] Models compared under cross-validation
- [ ] A metric that matches the goal
- [ ] An honest test
- [ ] A clear interpretation
- [ ] Assesses code
- [ ] A structured package
- [ ] Classes
- [ ] Tests
- [ ] command-line tool
- [ ] Notebook 
- [ ] Report
- [ ] AI_USAGE.md
- [ ] Presentation
- [ ] Practice defense


