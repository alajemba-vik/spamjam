"""Fixed facts used across the package: project paths and the target labels."""

from pathlib import Path

CONFIG_FILE = "config.yaml"


def find_project_root() -> Path:
    """Return the project folder: the one that contains config.yaml.

    All paths in config.yaml are relative to it, so they don't depend on the
    folder you run the code from.
    """
    # With `pip install -e .` the code runs from the repo:
    # /Users/user/spamjam/src/spamjam/constants.py -> parents[2] = /Users/user/spamjam
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / CONFIG_FILE).exists():
        return source_root
    # With a normal `pip install .` the code is copied into site-packages, so we
    # look in the current folder and its parents instead (e.g. running `spamjam`
    # from the project folder, or a notebook from notebooks/).
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / CONFIG_FILE).exists():
            return folder
    return source_root  # not found: the CLI reports it with a clear message


PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / CONFIG_FILE

# Target variable and its two labels
TARGET_COL = "is_spam"
HAM = 0
SPAM = 1
