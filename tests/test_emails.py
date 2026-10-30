"""Tests for reading many emails: folders, .eml files, .mbox files and labels."""

import mailbox
import re
from email.message import EmailMessage

import pytest

from spamjam.constants import HAM, SPAM, TARGET_COL
from spamjam.emails import (
    EMAIL_COL,
    ERROR_NO_EMAILS,
    TEXT_COL,
    features_table,
    read_emails,
)

SPAM_TEXT = "Get FREE money today!!! Click here to claim your $500."
HAM_TEXT = "Hi Claire, can we move tomorrow's call with the client to 3 pm?"


def make_eml(subject: str, body: str) -> bytes:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = "someone@example.com"
    message.set_content(body)
    return message.as_bytes()


@pytest.fixture
def labelled_folder(tmp_path):
    (tmp_path / "spam").mkdir()
    (tmp_path / "ham").mkdir()
    (tmp_path / "spam" / "offer.txt").write_text(SPAM_TEXT, encoding="utf-8")
    (tmp_path / "ham" / "call.eml").write_bytes(make_eml("Call", HAM_TEXT))
    (tmp_path / "ham" / "notes.pdf").write_bytes(b"not an email")  # ignored
    return tmp_path


def test_folders_give_the_labels(labelled_folder):
    emails = read_emails(labelled_folder)
    labels = dict(zip(emails[EMAIL_COL], emails[TARGET_COL]))
    assert labels == {"ham/call.eml": HAM, "spam/offer.txt": SPAM}


def test_eml_keeps_the_subject_and_body_but_not_the_headers(labelled_folder):
    emails = read_emails(labelled_folder)
    text = emails.loc[emails[EMAIL_COL] == "ham/call.eml", TEXT_COL].item()
    assert "Subject: Call" in text and HAM_TEXT in text
    assert "someone@example.com" not in text


def test_mbox_named_spam_is_labelled_spam(tmp_path):
    box = mailbox.mbox(tmp_path / "spam.mbox")
    box.add(make_eml("Offer", SPAM_TEXT))
    box.add(make_eml("Offer 2", SPAM_TEXT))
    box.close()
    emails = read_emails(tmp_path / "spam.mbox")
    assert len(emails) == 2
    assert (emails[TARGET_COL] == SPAM).all()


def test_unlabelled_emails_have_no_target_column(tmp_path):
    (tmp_path / "a.txt").write_text(SPAM_TEXT, encoding="utf-8")
    table = features_table(read_emails(tmp_path), ["word_freq_free", "char_freq_!"])
    assert list(table.columns) == ["word_freq_free", "char_freq_!"]


def test_labelled_emails_give_a_dataset(labelled_folder):
    table = features_table(read_emails(labelled_folder), ["word_freq_free"])
    assert list(table.columns) == ["word_freq_free", TARGET_COL]
    assert table[TARGET_COL].tolist() == [HAM, SPAM]


def test_only_folders_inside_the_given_one_give_labels(tmp_path):
    # A file called spam.txt, inside a project folder called "spam", is not
    # labelled: only the spam/ and ham/ folders inside the emails folder count.
    emails = tmp_path / "spam" / "emails"
    emails.mkdir(parents=True)
    (emails / "spam.txt").write_text(SPAM_TEXT, encoding="utf-8")
    assert read_emails(emails)[TARGET_COL].isna().all()


def test_a_folder_without_emails_is_rejected(tmp_path):
    error = ERROR_NO_EMAILS.format(path=tmp_path)
    with pytest.raises(ValueError, match=re.escape(error)):
        read_emails(tmp_path)
