from pathlib import Path

# We define a project root so that paths do not depend on the folder you run them from
# Path(__file__).resolve() returns /Users/user/spamjam/src/spamjam/constants.py
# parents[2] returns /Users/user/spamjam, which is the root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Target variable
TARGET_COL = "is_spam"
RANDOM_STATE = 42
