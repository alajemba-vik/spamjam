# spamjam
SpamJam is a machine learning project for email spam detection, featuring reproducible data pipelines, model comparison, cost-aware classification, automated testing, a Python command-line interface and a Streamlit page. To enable a more immersive learning experience, students have framed the problem around a fictional company GrandPied.


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


# Results

| | |
|---|---|
| Final model | Logistic regression (L1, `C`=1), with a log transform of the capital-letter counts and standard scaling |
| Model selection | Average precision 0.954 in 5-fold cross-validation on the training set (dummy baseline: 0.399). Tuned k-NN and linear SVM are tied with it; we keep the simplest |
| Decision threshold | 0.782, chosen on out-of-fold predictions to reach a precision of 0.95 on spam |
| Test set (used once) | Precision 0.950, recall 0.795: 14 of 506 legitimate emails flagged as spam, 69 of 336 spam missed |
| Second version, without the donor's words | Average precision 0.938 in cross-validation, threshold 0.82. Test set: precision 0.952, recall 0.711 (12 legitimate emails flagged, 97 spam missed). Same precision, but less spam caught, and it isn't fooled by an email that starts with "Hi George" |

The full analysis is in `notebooks/analysis.ipynb`. These scores are likely optimistic for GrandPied (see the limitations below).


# Limitations

The main ones:

- **One donor:** all the real emails come from one person, so words like `george` and `hp` carry a lot of weight on Spambase but would mean nothing at GrandPied. The second version, trained without them, shows what they are worth.
- **English only:** the 48 word features are English words, so French spam almost always gets through.
- **GrandPied's emails look like spam to it:** words a marketing company uses every day (`free`, `our`, `business`, `your`) push towards spam, so an ordinary email about a client order is flagged.
- **The 95 % precision depends on the mix:** it holds when about 40 % of the mail is spam, as in Spambase. With 10 % spam it would be about 76 %, so the threshold would have to be chosen again on GrandPied's emails.

The full list, with the numbers and examples behind each point, is in [`docs/limitations.md`](docs/limitations.md).


# Installation

Requires Python 3.10 to 3.14.

```bash
git clone https://github.com/alajemba-vik/spamjam.git
cd spamjam
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e .                   # the package, the `spamjam` command and everything it needs
```

This one command installs everything the project uses: the `spamjam` command, the Streamlit page, pytest, ruff and the Jupyter kernel for the notebooks. All versions are pinned in `pyproject.toml`.

With [uv](https://docs.astral.sh/uv/), `uv run spamjam ...` and `uv run pytest` work directly, without the steps above.


# Usage

Run the commands from the project folder (the one that contains `config.yaml`). Every setting (data paths, model, grid, thresholds) comes from `config.yaml`; `--config other.yaml` uses another settings file.

SpamJam trains two versions of the same model: one with all 57 features (the main one), and one without the donor's words (`george`, `hp`, ...; see `docs/limitations.md`). Each has its own threshold.

**Train** both versions on the training set and save them in `models/`. On the first run, it also downloads the data if needed and creates the train/test split. Run this first: `evaluate`, `predict` and the Streamlit page all use the saved models, and the models are not stored in git.

```bash
spamjam train
```

**Evaluate** both saved versions on the test set (scores and confusion matrix). The main version's results:

```bash
spamjam evaluate
```

```
== Model with all features ==
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

**Predict** one new email, either from its 57 feature values in a JSON file, or from its text:

```bash
spamjam predict --input examples/new_case.json
spamjam predict --text examples/new_email.txt
```

```
{"spam_probability": 0.9777, "is_spam": 1}
```

Add `--without-donor-words` to use the second version.

The model only sees the 57 features, never the text itself. With `--input`, the JSON file gives each value with its feature name (every feature is described in `docs/data_dictionary.md`); a missing, unknown or non-numeric value is rejected with a clear message instead of being guessed. With `--text`, SpamJam computes the 57 values from the email the way the Spambase documentation describes. That documentation doesn't say whether the headers were counted, so these values are close to what Spambase would give, not guaranteed identical. `--text` also takes an `.eml` file.

**Predict many emails at once:** `--text` also takes a folder of `.txt`/`.eml` emails or an `.mbox` file (the format most mail apps export to). It prints one line per email, or saves them with `--output`:

```bash
spamjam predict --text my_emails/ --output predictions.csv
```

**Build a dataset from your own emails:** `spamjam features` computes the 57 features of every email and saves them in the Spambase format. Emails in a `spam/` or `ham/` folder (or `spam.mbox`, `ham.mbox`) get that label in an `is_spam` column:

```bash
spamjam features grandpied_emails/ --output data/raw/grandpied.csv
```

This is how GrandPied could retrain SpamJam on its own emails: in a copy of `config.yaml`, set `data.path` to that file and give new `train_path`/`test_path`, then run `spamjam --config grandpied.yaml train`. The thresholds would then have to be chosen again in `notebooks/analysis.ipynb` (section Decision threshold), since they depend on the data. Real filters go further and learn their words from each user's own emails; see the end of `docs/limitations.md`.


**Try it in the browser** with the Streamlit page (run `spamjam train` first):

```bash
streamlit run src/spamjam/app.py
```

It opens a page where you either edit the feature values (the box starts with `examples/new_case.json`, so you can keep it, change some values or paste your own) or paste the text of an email. Next to it, it shows the spam probability, the decision at the threshold from `config.yaml`, and which features pushed the email towards spam or ham. A switch picks the second version, without the donor's words. It uses the same saved models and thresholds as `spamjam predict`.


# Tests and code quality

```bash
pytest        # 69 tests: data, pipeline (no leakage), training, metrics, threshold, prediction, features from text, reading many emails, both versions, command line, Streamlit page
ruff check .  # style (PEP 8), errors, import order, names and docstrings, notebooks included
```

The tests that use the command line or the Streamlit page train their own small models on a temporary split, so they never touch the real test set or your saved models, and they work before `spamjam train` has been run.


# Notebooks

1. `notebooks/data_exploration.ipynb`: the data, its quality, duplicates, skewed features, donor words, and the train/test split.
2. `notebooks/analysis.ipynb`: model selection with cross-validation, the decision threshold, the one-time test set evaluation, and the interpretation.

Open them with the Jupyter kernel of the virtual environment. In `analysis.ipynb`, the cell that tunes all models takes a few minutes (mostly the linear SVM); it prints its progress.


# Project structure

```
config.yaml              all settings: paths, seed, CV, grids, final model, thresholds
src/spamjam/
    data.py              DatasetLoader: download, remove duplicates, split
    features.py          preprocessing pipeline (log transform, scaling, dropping the donor's words)
    models.py            Trainer: tune with cross-validation, evaluate, save, load
    evaluate.py          metrics for a given threshold
    predict.py           prediction for one email read from JSON
    text.py              the 57 features computed from an email's text
    emails.py            reading many emails: folders, .eml and .mbox files, labels
    cli.py               the `spamjam` command
    app.py               the Streamlit page
    config.py            reads config.yaml
    constants.py         project paths and labels
tests/                   pytest tests
notebooks/               data exploration and analysis
results/                 charts saved by the notebooks, for the report and slides
examples/                example inputs for `spamjam predict` (JSON and text)
docs/data_dictionary.md  what every column of the dataset means
docs/limitations.md      what SpamJam can't do, with the numbers behind it
```


# Sources

- Data: Hopkins, M., Reeber, E., Forman, G. and Suermondt, J. (1999). *Spambase* [Dataset]. UCI Machine Learning Repository, CC BY 4.0. https://doi.org/10.24432/C53G6X
- Feature definitions (used in `src/spamjam/text.py`): the Spambase documentation on the UCI page above.
- Library: Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Published results we compare with (`docs/limitations.md`): Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer (the spam example); Shalizi, C., *Data Mining* (36-462), Carnegie Mellon University, homework 2: https://www.stat.cmu.edu/~cshalizi/dm/22/hw/02/hw-02.pdf
- How real filters are personal (`docs/limitations.md`): Apache SpamAssassin, `sa-learn` manual: https://manpages.debian.org/jessie/spamassassin/sa-learn.1p.en.html; Google Workspace, *An overview of Gmail's spam filters*: https://workspace.google.com/blog/identity-and-security/an-overview-of-gmails-spam-filters
- Margin of error: the Wilson score interval for a proportion (Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212).
- Use of AI tools: see `AI_USAGE.md`.


# Work Breakdown

https://github.com/users/alajemba-vik/projects/5/views/1


# Quick Checklist
- [x] A clean pipeline
- [x] Models compared under cross-validation
- [x] A metric that matches the goal
- [x] An honest test
- [x] A clear interpretation
- [x] Assesses code
- [x] A structured package
- [x] Classes
- [x] Tests
- [x] command-line tool
- [x] Notebook 
- [ ] Report
- [ ] AI_USAGE.md
- [ ] Presentation
- [ ] Practice defense


