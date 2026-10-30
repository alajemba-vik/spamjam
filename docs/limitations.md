# Limitations

SpamJam is a proof of concept built on the public Spambase dataset. This file
lists what it can't do, and why, with the numbers behind each point. It is the
one place where these details live: the README, the notebook and the report
link here.

Unless said otherwise, the numbers come from the test set (842 emails,
threshold 0.782) and are computed in `notebooks/analysis.ipynb`, section
Limitations. The example emails can be checked with
`spamjam predict --text FILE` or on the Streamlit page.


## 1. The real emails all come from one person

All the legitimate emails in Spambase come from the donor's own mailbox
(George, at HP Labs, around 1999). Words that describe him are among the
strongest signals the model uses: `george` has the largest weight by far
(about −8.2, towards a real email), followed by `hp`, `edu`, `cs`, `meeting`,
`85` and `lab`. In the training set, `hp` appears in about 40 % of real emails and
3 % of spam.

These words help on Spambase only because they identify the donor's emails.
Without them, cross-validation average precision drops from 0.954 to 0.938.
At GrandPied they mean nothing, so our scores are likely optimistic. The UCI
documentation itself says a general filter would have to "blind" these
words or use real emails from many people.

The published results we compare with (UCI, Hastie, Tibshirani and Friedman,
Shalizi at CMU) also train on all 57 features, so the comparison is fair, but
none of them removes these words either.


## 2. English only

The 48 word features are all English words (`free`, `money`, `remove`,
`business`, ...). A French email matches none of them, so the model only has
the punctuation (`!`, `$`, `(`, `;`, `[`, `#`) and the capital letters to go
on.

| Email | Spam probability | Verdict |
|---|---|---|
| Get FREE money today!!! Click here to claim your $500. | 100 % | quarantine |
| Gagnez de l'argent GRATUIT aujourd'hui !!! Cliquez ici pour recevoir vos 500 euros. | 38 % | inbox |
| Gagnez de l'argent gratuit aujourd'hui. Cliquez ici pour recevoir vos 500 euros. | 4 % | inbox |

French spam almost never reaches the threshold (78.2 %). French real emails
aren't flagged either, so no client email is lost, but the filter does
almost nothing for French mail. GrandPied would need French words, which
means new features and a model trained on its own French emails.


## 3. "95 % precision" only holds for Spambase's share of spam

On the test set, 95 % of the emails SpamJam flags really are spam. That
number depends on how much of the mail is spam: about 40 % in Spambase. The
model's own error rates don't change with the mix (it flags 2.8 % of real
emails and catches 79.5 % of spam), but with less spam, the few real emails
it flags make up a bigger part of what it flags:

| Share of spam in the mail | Precision |
|---|---|
| 40 % (Spambase) | 95 % |
| 20 % | 88 % |
| 10 % | 76 % |
| 5 % | 60 % |

We don't know GrandPied's share of spam, so the threshold would have to be
chosen again on GrandPied's own emails to keep 95 %.


## 4. The test results have a margin of error

The test set is small: 281 emails flagged, 506 real emails. With 95 %
confidence (Wilson interval):

- precision is between 91.8 % and 97.0 % (we report 95.0 %);
- the share of real emails flagged is between 1.7 % and 4.6 % (we report
  2.8 %, 14 of 506).

The threshold was chosen as the lowest one whose out-of-fold precision
reaches 0.95, so it sits right at the target. Getting 0.950 on the test set
is consistent with that choice, but on another sample the precision could
just as well be a little below 0.95.


## 5. The model only sees 57 fixed numbers

Spambase gives 57 numbers per email, not the email itself. The model never
sees the sender, the links, the headers, the attachments or the order of the
words, which real spam filters rely on heavily. It also can't learn new
words: a spam about "bitcoin" or "crypto" looks the same to it as an email
without those words.


## 6. It is easy to fool

Since a few words carry a lot of weight, a spammer can push an email towards
"real" by adding them, or avoid the 48 words altogether. For example:

| Email | Spam probability |
|---|---|
| Get FREE money today!!! Click here to claim your $500. | 100 % |
| Hi George, Get FREE money today!!! Click here to claim your $500. | 0 % |

Longer spam with many spam words resists this better: our example
`examples/new_email.txt` stays at 100 % with "Hi George," added. Image-only
spam, which has almost no text, would also get through.


## 7. Old data, and a split that ignores time

The emails were collected around 1999. Spam has changed a lot since then
(phishing, image-only spam, obfuscated words). The dataset has no dates, so
our split is random: the test set isn't "future" emails, while in real use
the model would always predict on newer emails than it was trained on. This
makes our test score more optimistic than real use.


## 8. Emails pasted on the Streamlit page are approximate

The page (and `spamjam predict --text`) computes the 57 numbers from a
pasted email with the formulas in the UCI documentation. The documentation
doesn't say whether the headers were counted. They probably were: every one
of the 4,601 emails has at least one capital letter (the minimum is 1 in all
three capital-letter columns, see `docs/data_dictionary.md`), while a pasted
email body may have none, a value the model never saw. So pasted emails get
values close to Spambase's, not guaranteed identical.


## 9. Choices GrandPied would revisit

- The 0.95 precision target is our choice. It isn't derived from what a lost
  client email actually costs GrandPied compared with a spam that gets
  through.
- We compared three models (logistic regression, k-NN, linear SVM). Naive
  Bayes, the classic spam filter, and tree-based models weren't tried.
- The "Pourquoi ?" chart on the Streamlit page shows each feature's share of
  the decision. Features that appear together (`857` and `415`, `labs` and
  `telnet`) share the weight between them, so a single bar can understate how
  much that group matters.


## 10. GrandPied's own emails look like spam to it

GrandPied is a marketing company, and the words its real emails use every day
are the words the model learned from spam: `free` (+0.65), `our` (+0.34),
`business` (+0.32), `credit`, `internet`, `your` and `you` all push towards
spam. Short emails between two colleagues show it:

| Email | Spam probability | Verdict |
|---|---|---|
| Hi Claire, can we move tomorrow's call with the client to 3 pm? Thanks, Paul | 0 % | inbox |
| Hi Claire, thanks for your order. Our team will send you the campaign report and the free trial details by email today. Best, Paul | 96 % | quarantine |
| Hi Claire, our new offer for your business: order before Friday and get free delivery on your next campaign. Best, Paul | 100 % | quarantine |

On Spambase, these words mostly come from spam, because the donor worked in a
research lab. At GrandPied, many real emails, including the ones to and from
clients, would be flagged.


## Risks of deploying it

- **Errors.** Even at 95 % precision, real emails get flagged (14 of 506 on
  the test set), and at GrandPied far more (see point 10). Each one may be a
  lost client inquiry, so flagged emails should go to a quarantine folder that
  people check, never be deleted.
- **Bias.** The model learned what a normal email looks like from one
  person's mailbox. Emails that don't look like his (French, marketing,
  written by people outside his field) are judged more harshly, while a spam
  that mentions "george" or "hp" gets through more easily.
- **Privacy.** The model reads the content of every email (words, characters,
  capital letters). Using it at GrandPied requires informing employees, clear
  rules on who can see quarantined emails, and compliance with privacy law
  (GDPR).


## What it means for GrandPied

Before real use, GrandPied would need to retrain the model on its own
labelled emails (French and English, marketing included), choose the
threshold again on its own share of spam, send flagged emails to quarantine
rather than delete them, and keep an eye on how many real emails get flagged.
If clients say their emails went unanswered, the threshold should be raised
so that fewer emails are flagged.
