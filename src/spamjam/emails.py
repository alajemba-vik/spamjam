"""Read many emails at once, to compute their features or predict them.

An email collection is either:

- a folder of ``.txt`` and ``.eml`` files (subfolders included), or
- an ``.mbox`` file, the format most mail apps export to.

If the emails sit in a folder called ``spam`` or ``ham`` (or the file is
``spam.mbox`` / ``ham.mbox``), that is their label. This is how GrandPied could
build a labelled dataset from its own emails and retrain SpamJam on it.

For ``.eml`` and ``.mbox`` emails we keep the subject and the text of the body,
not the technical headers (servers, dates, ids), which would add capital
letters and numbers that aren't part of the message.
"""

import mailbox
import re
from email import message_from_bytes, policy
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from spamjam.constants import HAM, SPAM, TARGET_COL
from spamjam.text import features_from_text

EMAIL_FILES = {".txt", ".eml"}
MBOX_FILE = ".mbox"
LABELS = {"ham": HAM, "spam": SPAM}
HTML_TAG = re.compile(r"<[^>]+>")

# Column with the file (or mbox position) each email comes from.
EMAIL_COL = "email"


def message_text(message: EmailMessage) -> str:
    """Return the subject and the body of an email message."""
    body = message.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body is not None else ""
    if body is not None and body.get_content_type() == "text/html":
        text = HTML_TAG.sub(" ", text)
    return f"Subject: {message['subject'] or ''}\n\n{text}"


def label_of(names: list[str]) -> int | None:
    """Return the label given by the first ``spam`` or ``ham`` name, if any."""
    for name in names:
        if name.lower() in LABELS:
            return LABELS[name.lower()]
    return None


def read_emails(path: Path) -> pd.DataFrame:
    """Return one row per email: where it comes from, its text and its label."""
    rows = []
    if path.is_file() and path.suffix == MBOX_FILE:
        for i, message in enumerate(mailbox.mbox(path, factory=None)):
            parsed = message_from_bytes(message.as_bytes(), policy=policy.default)
            label = label_of([path.stem])   # spam.mbox or ham.mbox
            rows.append((f"{path.name}#{i}", message_text(parsed), label))
    elif path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.suffix == ".txt":
                text = file.read_text(encoding="utf-8", errors="replace")
            elif file.suffix == ".eml":
                text = message_text(
                    message_from_bytes(file.read_bytes(), policy=policy.default)
                )
            else:
                continue
            # Only the folders inside `path` count, not the file's own name or
            # the folders above `path` (e.g. a "spam" project folder).
            folders = file.relative_to(path).parent.parts
            rows.append((str(file.relative_to(path)), text, label_of(list(folders))))
    else:
        raise ValueError(f"{path} is not a folder of emails or an .mbox file")
    if not rows:
        raise ValueError(f"No .txt, .eml or .mbox emails found in {path}")
    return pd.DataFrame(rows, columns=[EMAIL_COL, "text", TARGET_COL])


def features_table(emails: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return the features of every email, with its label when it has one.

    The columns are the Spambase ones, in the same order, so the table can be
    used as a dataset to train SpamJam (``data.path`` in config.yaml).
    """
    table = pd.DataFrame(
        [features_from_text(text, feature_names) for text in emails["text"]],
        columns=feature_names,
    )
    if emails[TARGET_COL].notna().all():
        table[TARGET_COL] = emails[TARGET_COL].astype(int)
    return table
