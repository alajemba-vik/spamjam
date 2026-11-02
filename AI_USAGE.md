# AI usage

This file declares every use of generative AI in SpamJam, as the project brief
requires: which tools, for what, and what we changed. It follows the usual
structure of AI disclosure statements: tools, uses by area, verification, and
responsibility.

## Summary

We used two AI assistants throughout the project, for explanations, code,
reviews and writing. Every output was read, tested and, where needed, corrected
or rejected by us. We made the project's decisions ourselves, and we can
explain every part of the code, the notebooks, the report and the slides.

## Tools

| Tool | Provider | How we accessed it |
|---|---|---|
| ChatGPT | OpenAI | Chat in the browser |
| Claude | Anthropic | The Claude app, with access to our repository, so it could read files, run the tests and propose changes |

## Uses by area

| Area | What the AI did | What we did and changed |
|---|---|---|
| Planning | ChatGPT checked our split of the work into tasks and issues. | We wrote the issues, assigned them and decided the order. |
| Learning | Both explained concepts before we used them (cross-validation, average precision, out-of-fold predictions, the one-standard-error rule, Pipelines and ColumnTransformer). | We decided which methods to use and checked them against the course labs. |
| Code | ChatGPT suggested fixes for scikit-learn errors (`GridSearchCV` step names, `BaseEstimator`). Claude wrote or refactored parts of the package: the command line (`cli.py`), the second model version, `text.py`, `emails.py`, the Streamlit page (`app.py`), and the constants refactor. | We asked for each change, reviewed every diff before committing, asked for corrections (for example removing hard-coded values from the tests) and rejected changes we disagreed with. |
| Tests | ChatGPT drafted tests for the decision threshold and the predicted probabilities. Claude wrote the tests for the command line, the Streamlit page, `text.py` and `emails.py`, and the fixture that trains small models for the tests. | We ran the tests on our machines (macOS and Windows), found the failures on a fresh clone and on Windows, and checked what each test verifies. |
| Notebooks | Claude reviewed the notebooks, fixed errors in the text and the code (the threshold section, the coefficient chart's feature order) and wrote the sections on the second version and on the margin of error. | We ran both notebooks from top to bottom ourselves and checked every printed number against the text. |
| Debugging | ChatGPT helped resolve Git merge conflicts and a corrupted notebook. Claude diagnosed the fresh-clone test failures and the altair / Python 3.10 conflict. | Sheila found the altair problem on Windows; we chose to pin altair 6.2.2 instead of dropping Python 3.10. |
| Reviews | Claude reviewed our pull requests and audited the whole project several times (correctness of numbers, contradictions, code quality). | We decided which findings to act on and made the final call on each pull request. |
| Documentation | Claude drafted parts of the README, `docs/data_dictionary.md` and `docs/limitations.md`, and checked each claim against the data. | We asked for limitations it had missed (for example that the model only understands English), corrected wording, and removed claims we could not support. |
| Report and slides | Claude generated the Word report, its appendix and the PowerPoint slides from our results, and rebuilt them after each change. | We chose the content and the structure, corrected errors (for example two missing standard deviations in Table 2) and asked for the GrandPied style. |
| Git | Claude drafted commit messages and pull request descriptions. | We committed and pushed ourselves, under our own names. |

## What we did not delegate

- The choice of dataset, the GrandPied framing and the cost of each kind of
  mistake.
- The decisions: the metric, the target precision, keeping the simplest of the
  tied models, training a second version without the donor's words (after
  asking our teacher), and keeping the GrandPied story.
- Running the notebooks, the tests and the fresh installs, and submitting.

## How we checked the AI's work

- Every number in the README, the notebooks, `docs/limitations.md`, the report
  and the slides was recomputed from the code or the data.
- The tests pass on a fresh clone, before training, on Python 3.11, 3.13 and
  3.14 (macOS); Sheila also ran the project on Windows.
- `ruff check .` passes.
- AI suggestions that were wrong were corrected or dropped (for example an
  outdated coefficient chart in the report, an overclaim about the threshold
  "generalising", and a requirement that would have locked out Python 3.10).

## Responsibility

We are responsible for everything we submit. We read and understood all of
it, and we can explain any line of the code and any number in the report.
