"""Turn the text of an email into the features the model was trained on.

The Spambase documentation says how each feature was computed from an email:

- word_freq_WORD: 100 × (times WORD appears) / (number of words), where a word
  is any run of letters and digits;
- char_freq_CHAR: 100 × (times CHAR appears) / (number of characters);
- capital_run_length_*: the average length, the longest length and the total
  length of the runs of capital letters ("FREE MONEY" has runs of 4 and 5).

It doesn't say whether the email headers were counted, or exactly what counts
as a character, so the values for a pasted email are close to what Spambase
would give, not guaranteed identical.
"""

import re

WORD_PREFIX = "word_freq_"
CHAR_PREFIX = "char_freq_"
CAPITAL_PREFIX = "capital_run_length_"

WORD = re.compile(r"[A-Za-z0-9]+")
CAPITAL_RUN = re.compile(r"[A-Z]+")


def percentage(count: int, total: int) -> float:
    """Return 100 × count / total, or 0 for an empty email."""
    return 100 * count / total if total else 0.0


def features_from_text(text: str, feature_names: list[str]) -> dict:
    """Return the value of each named feature for one email's text."""
    # Windows ends lines with "\r\n", Mac and Linux with "\n". Counting the
    # extra "\r" would change the character frequencies, so we remove it.
    text = text.replace("\r\n", "\n")
    words = [word.lower() for word in WORD.findall(text)]
    runs = [len(run) for run in CAPITAL_RUN.findall(text)]
    capitals = {
        "average": sum(runs) / len(runs) if runs else 0.0,
        "longest": max(runs, default=0),
        "total": sum(runs),
    }

    features = {}
    for name in feature_names:
        if name.startswith(WORD_PREFIX):
            word = name.removeprefix(WORD_PREFIX)
            features[name] = percentage(words.count(word), len(words))
        elif name.startswith(CHAR_PREFIX):
            char = name.removeprefix(CHAR_PREFIX)
            features[name] = percentage(text.count(char), len(text))
        elif name.startswith(CAPITAL_PREFIX):
            features[name] = capitals[name.removeprefix(CAPITAL_PREFIX)]
        else:
            raise ValueError(f"Don't know how to compute {name} from text")
    return features
