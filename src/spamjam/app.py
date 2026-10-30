"""Streamlit page to try SpamJam on one email, in GrandPied's colours.

    streamlit run src/spamjam/app.py

It uses the model saved by `spamjam train` and the threshold from config.yaml,
so it gives the same answers as `spamjam predict`.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from spamjam.config import load_config
from spamjam.constants import CONFIG_PATH, PROJECT_ROOT, SPAM
from spamjam.models import Trainer, version_settings
from spamjam.predict import features_to_row, predict_email
from spamjam.text import features_from_text

# GrandPied's colours (the same as in .streamlit/config.toml and the slides).
FOREST, GREEN, PINK, RASPBERRY = "#173F2E", "#1A7F4E", "#EB8DB0", "#B03A68"

# Text shown on the page.
PAGE_TITLE = "SpamJam · GrandPied"
KICKER = "GrandPied · Service informatique"
TITLE = "SpamJam"
TAGLINE = "Le tri du courrier, sans perdre un client."
HELLO = "Bonjour !"
INTRO = (
    "Pick an email and SpamJam tells you whether it is spam. At GrandPied a lost "
    "client email costs more than a spam that gets through, so the model only "
    "flags an email when it is quite sure, and flagged emails go to quarantine, "
    "never to the bin."
)
CHOICE_LABEL = "Which email?"
CHOICE_FEATURES = "Edit the feature values"
CHOICE_TEXT = "Paste an email"
HELP_FEATURES = (
    "The model reads 57 numbers per email: how often some words and characters "
    "appear, and the runs of capital letters. The box holds the example email "
    "(`examples/new_case.json`): keep it, change some values, or paste your own. "
    "Every feature is described in `docs/data_dictionary.md`."
)
HELP_TEXT = (
    "SpamJam counts the words, characters and capital letters the way the "
    "Spambase data was built. English emails only: the model's words are all "
    "English. Paste the whole email, subject included. The "
    "dataset doesn't say exactly what it counted, so the numbers are close to "
    "Spambase's, not identical."
)
LABEL_FEATURES = "Feature values (JSON)"
LABEL_TEXT = "Email text"
PLACEHOLDER_TEXT = (
    "Paste an email here, subject included.\n\n"
    "Try this: paste a short spam, like \"Get FREE money today!!! Click here to "
    "claim your $500.\", then add \"Hi George,\" at the top. The verdict flips, "
    "because all the real emails in Spambase were sent to George at HP Labs, so the "
    "model learned that his name means a real email. That's why GrandPied would "
    "have to retrain it on its own emails."
)
LABEL_COMPUTED = "The 57 numbers SpamJam computed from this email"
MSG_PASTE = "Paste an email above, then press Check this email."
BUTTON_CHECK = "Check this email"
VERDICT_HEADER = "Le verdict"
WHY_HEADER = "Pourquoi ?"
VERDICT_SPAM = ("En quarantaine", "This looks like spam. It would wait in quarantine, "
                "where anyone can still rescue it.")
VERDICT_HAM = ("Livré", "This looks like a real email. It would go straight to "
               "the inbox.")
MSG_PROBABILITY = "Spam probability: {probability:.1%} (flagged from {threshold:.1%})"
MSG_EXPLANATION = (
    "The threshold of {threshold} was chosen so that about 95 % of the emails "
    "SpamJam flags really are spam."
)
MSG_WHY = (
    "The features that pushed this email the most. Pink pushes towards spam, "
    "green towards a real email: each bar is the feature's weight in the model "
    "times its value for this email."
)
INFO_TITLE = "Comment ça marche"
INFO_TEXT = (
    "- **Model:** {name}, trained by `spamjam train`, in two versions: with all "
    "57 features, and without the donor's words (switch on the left)\n"
    "- **Thresholds:** {threshold} and {threshold_no_donor}, from `config.yaml`\n"
    "- **On the test set:** about 95 % of flagged emails were spam with both "
    "versions (14 and 12 of 506 real emails flagged). The 95 % holds when about "
    "40 % of the mail is spam, as in Spambase; with less spam it is lower\n"
    "- **English only:** the model's words are all English, so French spam "
    "almost always gets through\n"
    "- Same answers as `spamjam predict`\n"
    "- All the limitations: `docs/limitations.md`"
)
FOOTER = (
    "Proof of concept trained on the public Spambase dataset, whose real emails "
    "all come from one person. Before real use, retrain it on GrandPied's own "
    "emails. Fait avec soin par l'équipe IT de GrandPied."
)
ERROR_NO_MODEL = "No model at {path}. Run `spamjam train` first, then reload."
LABEL_NO_DONOR = "Without the donor's words"
HELP_NO_DONOR = (
    "Spambase's real emails all come from one person, George at HP Labs, so "
    "words like george or hp mean \"real email\" to the main model. This version "
    "was trained without them. Try it on an email that starts with \"Hi George\"."
)
ERROR_BAD_INPUT = "Cannot predict: {error}"

# Number of features shown in the "Pourquoi ?" chart.
TOP_FEATURES = 10

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stRadio, .stCaption {{
    font-family: 'Poppins', sans-serif;
}}
.st-key-hero {{
    background: {FOREST}; color: white; border-radius: 18px;
    padding: 1.1rem 1.8rem 1.5rem; margin-bottom: 1.2rem;
}}
/* Streamlit pulls markdown up by 1rem; inside the banner we want even spacing */
.st-key-hero [data-testid="stMarkdownContainer"],
.st-key-hero [data-testid="stColumn"] {{ margin-bottom: 0; }}
/* Dialogs open in the middle of the screen instead of the top */
[data-testid="stDialog"] {{ align-items: center; }}
.st-key-info {{ margin-left: auto; }}
.st-key-info button {{ color: white; }}
.st-key-info button:hover {{ color: {PINK}; }}
.gp-kicker {{ color: {PINK}; font-size: 0.85rem; letter-spacing: 0.08em;
    text-transform: uppercase; font-weight: 600; }}
.st-key-hero h1 {{ color: white; font-size: 2.6rem; font-weight: 700;
    margin: 0.2rem 0 0; padding: 0; }}
.gp-tagline {{ color: #C9D9CF; font-size: 1.05rem; margin-top: 0.2rem; }}
.gp-sticker {{ display: inline-block; background: {PINK}; color: {FOREST};
    font-weight: 700; padding: 0.15rem 0.9rem; border-radius: 999px;
    transform: rotate(-3deg); margin: 0.6rem 0 0.4rem; }}
.gp-verdict {{ border-radius: 16px; padding: 1.1rem 1.4rem; margin: 0.4rem 0 0.6rem; }}
.gp-verdict h3 {{ margin: 0 0 0.2rem; padding: 0; font-weight: 700; }}
.gp-spam {{ background: #FBE8EF; border: 2px solid {PINK}; }}
.gp-spam h3 {{ color: {RASPBERRY}; }}
.gp-ham {{ background: #EAF3EC; border: 2px solid #A8C9B0; }}
.gp-ham h3 {{ color: {GREEN}; }}
.stTextArea textarea {{ font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.85rem; }}
.gp-footer {{ color: #5E6B64; font-size: 0.8rem; margin-top: 2rem;
    border-top: 1px solid #E6EBE7; padding-top: 0.8rem; }}
</style>
"""


def parse_features(text: str) -> dict:
    """Return one email's feature values from JSON text."""
    features = json.loads(text)
    if not isinstance(features, dict):
        raise ValueError("The input must be one JSON object: {feature: value, ...}.")
    return features


def feature_contributions(model: Pipeline, features: dict) -> pd.Series:
    """Return how much each feature pushes this email towards spam.

    For a linear model, the score of an email is the intercept plus the sum of
    weight × scaled value over the features, so each product is that feature's
    share of the decision. Positive means towards spam, negative towards ham.
    """
    row = features_to_row(features, list(model.feature_names_in_))
    # Every step but the model: the scaling, and removing the donor's words for
    # the second version.
    scaled = np.asarray(model[:-1].transform(row))[0]
    names = [
        name.split("__", 1)[1]   # drop the "capital__" / "other__" prefix
        for name in model["preprocess"].get_feature_names_out()
    ]
    return pd.Series(model["model"].coef_[0] * scaled, index=names)


@st.cache_resource
def load_model(model_path: Path, config_path: Path) -> Pipeline:
    """Load the saved model once and keep it while the page is open."""
    trainer = Trainer.from_config(config_path=config_path)
    return trainer.load(model_path)


def why_chart(contributions: pd.Series) -> alt.Chart:
    """Return a bar chart of the strongest contributions, pink for spam."""
    top = contributions[contributions.abs().sort_values(ascending=False).index]
    data = top.head(TOP_FEATURES).rename_axis("feature").reset_index(name="push")
    data["direction"] = data["push"].map(
        lambda push: "towards spam" if push > 0 else "towards a real email"
    )
    return alt.Chart(data).mark_bar(cornerRadius=4).encode(
        x=alt.X("push:Q", title="push"),
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=250)),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["towards spam", "towards a real email"],
                            range=[PINK, GREEN]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
    )


@st.dialog(INFO_TITLE)
def show_info(config: dict) -> None:
    """Explain how the page works, in a dialog opened from the banner."""
    st.markdown(INFO_TEXT.format(
        name=config["model"]["name"],
        threshold=version_settings(config)["threshold"],
        threshold_no_donor=version_settings(config, True)["threshold"],
    ))


def banner(config: dict) -> None:
    """Show the green GrandPied banner, with an info button at the top right."""
    with st.container(key="hero", gap=None):
        kicker, info = st.columns([12, 1], vertical_alignment="center")
        kicker.markdown(f'<div class="gp-kicker">👣 {KICKER}</div>',
                        unsafe_allow_html=True)
        if info.button("", icon=":material/info:", type="tertiary", key="info",
                       help=INFO_TITLE):
            show_info(config)
        st.markdown(f'<h1>{TITLE}</h1><div class="gp-tagline">{TAGLINE}</div>',
                    unsafe_allow_html=True)


def sticker(text: str) -> None:
    """Show a section title as a pink GrandPied sticker."""
    st.markdown(f'<span class="gp-sticker">{text}</span>', unsafe_allow_html=True)


def read_input(model: Pipeline, example_path: Path) -> dict:
    """Show the input box chosen with the radio button and return the features.

    Each box sits in a form with a button, so the email is only sent to the
    model when you press it (Ctrl+Enter or Cmd+Enter also work). Stops the page
    while the email box is empty.
    """
    choice = st.radio(CHOICE_LABEL, [CHOICE_FEATURES, CHOICE_TEXT], horizontal=True)
    if choice == CHOICE_FEATURES:
        st.caption(HELP_FEATURES)
        with st.form("features_form", border=False):
            example = example_path.read_text(encoding="utf-8")
            values = st.text_area(LABEL_FEATURES, example, height=320)
            st.form_submit_button(BUTTON_CHECK, type="primary")
        return parse_features(values)

    st.caption(HELP_TEXT)
    with st.form("text_form", border=False):
        text = st.text_area(LABEL_TEXT, height=320, placeholder=PLACEHOLDER_TEXT)
        st.form_submit_button(BUTTON_CHECK, type="primary")
    if not text.strip():
        st.info(MSG_PASTE)
        st.stop()
    features = features_from_text(text, list(model.feature_names_in_))
    with st.expander(LABEL_COMPUTED):
        st.json({name: round(value, 3) for name, value in features.items()})
    return features


def main() -> None:
    """Draw the page: the email on the left, the verdict on the right."""
    config = load_config(CONFIG_PATH)

    st.set_page_config(page_title=PAGE_TITLE, page_icon="👣", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    banner(config)

    left, right = st.columns(2, gap="large")
    with left:
        sticker(HELLO)
        st.write(INTRO)
        without_donor_words = st.toggle(LABEL_NO_DONOR, help=HELP_NO_DONOR)
        settings = version_settings(config, without_donor_words)
        threshold = settings["threshold"]
        if not settings["path"].exists():
            st.error(ERROR_NO_MODEL.format(path=settings["path"]))
            st.stop()
        model = load_model(settings["path"], CONFIG_PATH)
        try:
            features = read_input(model, PROJECT_ROOT / config["app"]["example_path"])
            result = predict_email(model, features, threshold)
        except ValueError as error:
            st.error(ERROR_BAD_INPUT.format(error=error))
            st.stop()

    with right:
        sticker(VERDICT_HEADER)
        is_spam = result["is_spam"] == SPAM
        title, text = VERDICT_SPAM if is_spam else VERDICT_HAM
        style = "gp-spam" if is_spam else "gp-ham"
        st.markdown(f'<div class="gp-verdict {style}"><h3>{title}</h3>{text}</div>',
                    unsafe_allow_html=True)
        st.progress(result["spam_probability"], text=MSG_PROBABILITY.format(
            probability=result["spam_probability"], threshold=threshold))
        st.caption(MSG_EXPLANATION.format(threshold=threshold))

        sticker(WHY_HEADER)
        st.write(MSG_WHY)
        st.altair_chart(why_chart(feature_contributions(model, features)),
                        use_container_width=True)

    st.markdown(f'<div class="gp-footer">{FOOTER}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
