"""Metrics used to evaluate the spam classifier."""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from spamjam.constants import HAM, SPAM

# Name of the confusion matrix in the dictionary returned by compute_metrics().
CONFUSION_MATRIX = "confusion_matrix"


def compute_metrics(
    y_true: pd.Series, spam_probability: list[float], threshold: float
) -> dict:
    """Return the classification metrics for predicted spam probabilities.

    An email is labelled spam when its probability is at least ``threshold``.
    Precision, recall, F1 and the confusion matrix depend on that threshold;
    ROC-AUC and average precision use the probabilities directly.
    """
    y_pred = [SPAM if p >= threshold else HAM for p in spam_probability]
    return {
        # zero_division=0: if nothing is predicted as spam (e.g. a dummy model),
        # precision is 0 instead of a division-by-zero warning.
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, spam_probability),
        "average_precision": average_precision_score(y_true, spam_probability),
        # [[true ham, false spam], [missed spam, true spam]]
        CONFUSION_MATRIX: confusion_matrix(y_true, y_pred, labels=[HAM, SPAM]),
    }
