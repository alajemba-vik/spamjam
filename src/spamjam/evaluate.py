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

# Names of the metrics in the dictionary returned by compute_metrics().
PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
ROC_AUC = "roc_auc"
AVERAGE_PRECISION = "average_precision"
SCORES = [PRECISION, RECALL, F1, ROC_AUC, AVERAGE_PRECISION]
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
        PRECISION: precision_score(y_true, y_pred, zero_division=0),
        RECALL: recall_score(y_true, y_pred, zero_division=0),
        F1: f1_score(y_true, y_pred, zero_division=0),
        ROC_AUC: roc_auc_score(y_true, spam_probability),
        AVERAGE_PRECISION: average_precision_score(y_true, spam_probability),
        # [[true ham, false spam], [missed spam, true spam]]
        CONFUSION_MATRIX: confusion_matrix(y_true, y_pred, labels=[HAM, SPAM]),
    }
