"""Tests for computing the features from the text of an email."""

import pytest

from spamjam.constants import WORD_PREFIX
from spamjam.text import ERROR_UNKNOWN_FEATURE, features_from_text

SPAMBASE_WORDS = 48   # word features in Spambase


def test_word_frequency_is_a_percentage_of_the_words():
    # 3 words, 2 of them "free" (any case): 100 * 2 / 3
    features = features_from_text("FREE money free", ["word_freq_free"])
    assert features["word_freq_free"] == pytest.approx(66.667, abs=0.001)


def test_a_word_is_a_run_of_letters_and_digits():
    # "e-mail" is two words, "e" and "mail", so "email" is not found.
    features = features_from_text("e-mail", ["word_freq_mail", "word_freq_email"])
    assert features == {"word_freq_mail": 50.0, "word_freq_email": 0.0}


def test_char_frequency_is_a_percentage_of_the_characters():
    features = features_from_text("Win!!", ["char_freq_!"])
    assert features["char_freq_!"] == 40.0


def test_capital_runs():
    names = ["capital_run_length_average", "capital_run_length_longest",
             "capital_run_length_total"]
    features = features_from_text("FREE MONEY now", names)
    assert features == {"capital_run_length_average": 4.5,
                        "capital_run_length_longest": 5,
                        "capital_run_length_total": 9}


def test_windows_line_endings_give_the_same_values():
    names = ["char_freq_!", "word_freq_free"]
    mac = features_from_text("Free!\nWin!", names)
    windows = features_from_text("Free!\r\nWin!", names)
    assert windows == mac


def test_a_french_email_matches_no_word_feature(real_df):
    # The 48 word features are English words, so the model can't read French
    # (see docs/limitations.md). If this fails, update that file.
    word_features = [name for name in real_df.columns if name.startswith(WORD_PREFIX)]
    french_spam = ("Gagnez de l'argent GRATUIT aujourd'hui !!! "
                   "Cliquez ici pour recevoir vos 500 euros.")
    features = features_from_text(french_spam, word_features)
    assert len(word_features) == SPAMBASE_WORDS
    assert set(features.values()) == {0}


def test_empty_email_gives_zeros():
    features = features_from_text("", ["word_freq_free", "char_freq_$",
                                       "capital_run_length_longest"])
    assert set(features.values()) == {0}


def test_unknown_feature_is_rejected():
    error = ERROR_UNKNOWN_FEATURE.format(name="word_count")
    with pytest.raises(ValueError, match=error):
        features_from_text("hello", ["word_count"])
