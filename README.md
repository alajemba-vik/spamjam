# spamjam
SpamJam is a machine learning project for email spam detection, featuring reproducible data pipelines, model comparison, cost-aware classification, automated testing, and a Python command-line interface. To enable a more immersive learning experience, students have framed the problem around a fictional company GrandPied.


# Problem
GrandPied is a marketing company that relies heavily on email for its daily operations. Recently, its employees have been getting more emails including unsolicited ads, spam and harmful messages.
However, blocking legitimate emails can also be an issue since this could result in a missed client inquiry, or even work a lost business opportunity.
To address this challenge, GrandPied's IT department is exploring SpamJam, an ML-based spam detection system.
The goal is to achieve an appropriate balance between identifying unwanted messages and protecting important business communications.

**Scope:** SpamJam is a proof of concept. We build and evaluate the full pipeline on the public Spambase dataset, whose legitimate emails all come from one person (George, at HP Labs, around 1999). Before real use, GrandPied would retrain the model on its own labelled emails with the same code.
- **Target**: Whether an email is spam or ham (legitimate), 1 and 0 respectively.
- **Task**: Supervised binary classification where we help IT admins identify unwanted emails while protecting legitimate business communication
 - **Cost**: GrandPied considers false positives (legitimate emails incorrectly classified as spam) more costly than false negatives (spam incorrectly classified as legitimate email) because it relies heavily on receiving client emails. Although false negatives allow unwanted messages through, GrandPied accepts this trade-off.


# Data
We use the [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) from the UCI Machine Learning Repository (Hopkins, Reeber, Forman and Suermondt, 1999), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It contains 4,601 emails described by 57 numeric features (word and character frequencies, and runs of capital letters).

**Limitation:** some features describe the dataset's donor rather than spam (e.g. `george`, `hp`, `650`). They help on Spambase but would not on GrandPied's emails, so our scores are likely optimistic for GrandPied. See `notebooks/analysis.ipynb` for the comparison with and without them.


# Work Breakdown

https://github.com/users/alajemba-vik/projects/5/views/1


# Quick Checklist
- [ ] A clean pipeline
- [ ] Models compared under cross-validation
- [ ] A metric that matches the goal
- [ ] An honest test
- [ ] A clear interpretation
- [ ] Assesses code
- [ ] A structured package
- [ ] Classes
- [ ] Tests
- [ ] command-line tool
- [ ] Notebook 
- [ ] Report
- [ ] AI_USAGE.md
- [ ] Presentation
- [ ] Practice defense


