"""Feature engineering and preprocessing pipeline for SpamJam."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from spamjam.constants import CAPITAL_PREFIX

# "^" means "starts with": picks every column whose name starts with
# "capital_run_length_" (the 3 capital-letter counts).
CAPITAL_PATTERN = rf"^{CAPITAL_PREFIX}"

# "(?!...)" means "NOT followed by": picks every column that does NOT start with
# "capital_run_length_", i.e. all the word and character frequencies.
# With these patterns we never have to type a list of 57 column names.
OTHER_PATTERN = rf"^(?!{CAPITAL_PREFIX})"


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer that scales every feature.

    Capital-letter counts are very skewed, so they are log-transformed
    before scaling. All other features are scaled.
    """
    # For the 3 capital-letter counts: log -> scale.
    # ORDER MATTERS: log first (to pull in extreme values), then scale (so all
    # features have a similar size).
    capital_pipeline = Pipeline([
        # Let's log-transform the capital-letter counts.
        # These counts are very skewed: one email has a run of 9,989 capitals
        # while 99% are under 669. The log squeezes big values
        # (9,989 -> 9.2, 669 -> 6.5) so a few extreme emails don't dominate.
        # "1 +" because log(0) is undefined. feature_names_out keeps the column
        # names, so we can still label the model's weights later.
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    # No log for the other features: these are percentages with a limited range,
    # so they don't need it.
    other_pipeline = Pipeline([
        ("scale", StandardScaler()),
    ])
    # make_column_selector picks the columns by the name regex patterns.
    return ColumnTransformer([
        ("capital", capital_pipeline, make_column_selector(pattern=CAPITAL_PATTERN)),
        ("other", other_pipeline, make_column_selector(pattern=OTHER_PATTERN)),
    ])


# THE FULL PIPELINE: ColumnTransformer -> model.
# "model" is a parameter so we can plug in ANY model (logistic regression,
#  k-NN, SVM) with exactly the same preparation: that makes the comparison fair.
def build_pipeline(
    model: BaseEstimator, drop_columns: list[str] | None = None
) -> Pipeline:
    """Return the full pipeline: preprocessing, then the model.

    With ``drop_columns`` (e.g. the donor's words), the pipeline first removes
    those columns. It still takes all 57 features as input, so a new email is
    given the same way to both versions of the model.
    """
    steps = [
        ("preprocess", build_preprocessor()),
        ("model", model),
    ]
    if drop_columns:
        drop = ColumnTransformer(
            [("drop", "drop", drop_columns)],
            remainder="passthrough",
            verbose_feature_names_out=False,   # keep the column names as they are
        ).set_output(transform="pandas")       # the next step selects columns by name
        steps.insert(0, ("drop", drop))
    return Pipeline(steps)
