# Data dictionary

Spambase dataset, UCI Machine Learning Repository, id 94
(Hopkins, Reeber, Forman and Suermondt, 1999, CC BY 4.0):
https://archive.ics.uci.edu/dataset/94/spambase

The definitions below follow the dataset's documentation. The ranges are the
minimum and maximum in `data/raw/spambase.csv` (4,601 emails, before
removing duplicates).

## Overview

| Group | Columns | Definition |
|---|---|---|
| Word frequencies | 48 `word_freq_*` | 100 × (number of times the word appears in the email) / (total number of words in the email). A word is any string of letters and digits bounded by other characters or the end of the text. |
| Character frequencies | 6 `char_freq_*` | 100 × (number of times the character appears) / (total number of characters in the email). |
| Capital letters | 3 `capital_run_length_*` | Statistics on the uninterrupted sequences of capital letters. |
| Target | `is_spam` | 1 = spam, 0 = ham. |

There are no missing values. The dataset contains these features only, not the
text of the emails.

The legitimate emails come from one person's mailbox at HP Labs, so some words
describe the donor rather than spam (`george`, `hp`, `hpl`, `650`, `857`,
`415`, `labs`, `telnet`, `technology`). See `notebooks/analysis.ipynb`.

## Columns

| Column | Type | Range | Description |
|---|---|---|---|
| `word_freq_make` | float, % | 0 – 4.54 | Percentage of the words in the email that are "make". |
| `word_freq_address` | float, % | 0 – 14.28 | Percentage of the words in the email that are "address". |
| `word_freq_all` | float, % | 0 – 5.1 | Percentage of the words in the email that are "all". |
| `word_freq_3d` | float, % | 0 – 42.81 | Percentage of the words in the email that are "3d". |
| `word_freq_our` | float, % | 0 – 10 | Percentage of the words in the email that are "our". |
| `word_freq_over` | float, % | 0 – 5.88 | Percentage of the words in the email that are "over". |
| `word_freq_remove` | float, % | 0 – 7.27 | Percentage of the words in the email that are "remove". |
| `word_freq_internet` | float, % | 0 – 11.11 | Percentage of the words in the email that are "internet". |
| `word_freq_order` | float, % | 0 – 5.26 | Percentage of the words in the email that are "order". |
| `word_freq_mail` | float, % | 0 – 18.18 | Percentage of the words in the email that are "mail". |
| `word_freq_receive` | float, % | 0 – 2.61 | Percentage of the words in the email that are "receive". |
| `word_freq_will` | float, % | 0 – 9.67 | Percentage of the words in the email that are "will". |
| `word_freq_people` | float, % | 0 – 5.55 | Percentage of the words in the email that are "people". |
| `word_freq_report` | float, % | 0 – 10 | Percentage of the words in the email that are "report". |
| `word_freq_addresses` | float, % | 0 – 4.41 | Percentage of the words in the email that are "addresses". |
| `word_freq_free` | float, % | 0 – 20 | Percentage of the words in the email that are "free". |
| `word_freq_business` | float, % | 0 – 7.14 | Percentage of the words in the email that are "business". |
| `word_freq_email` | float, % | 0 – 9.09 | Percentage of the words in the email that are "email". |
| `word_freq_you` | float, % | 0 – 18.75 | Percentage of the words in the email that are "you". |
| `word_freq_credit` | float, % | 0 – 18.18 | Percentage of the words in the email that are "credit". |
| `word_freq_your` | float, % | 0 – 11.11 | Percentage of the words in the email that are "your". |
| `word_freq_font` | float, % | 0 – 17.1 | Percentage of the words in the email that are "font". |
| `word_freq_000` | float, % | 0 – 5.45 | Percentage of the words in the email that are "000". |
| `word_freq_money` | float, % | 0 – 12.5 | Percentage of the words in the email that are "money". |
| `word_freq_hp` | float, % | 0 – 20.83 | Percentage of the words in the email that are "hp". |
| `word_freq_hpl` | float, % | 0 – 16.66 | Percentage of the words in the email that are "hpl". |
| `word_freq_george` | float, % | 0 – 33.33 | Percentage of the words in the email that are "george". |
| `word_freq_650` | float, % | 0 – 9.09 | Percentage of the words in the email that are "650". |
| `word_freq_lab` | float, % | 0 – 14.28 | Percentage of the words in the email that are "lab". |
| `word_freq_labs` | float, % | 0 – 5.88 | Percentage of the words in the email that are "labs". |
| `word_freq_telnet` | float, % | 0 – 12.5 | Percentage of the words in the email that are "telnet". |
| `word_freq_857` | float, % | 0 – 4.76 | Percentage of the words in the email that are "857". |
| `word_freq_data` | float, % | 0 – 18.18 | Percentage of the words in the email that are "data". |
| `word_freq_415` | float, % | 0 – 4.76 | Percentage of the words in the email that are "415". |
| `word_freq_85` | float, % | 0 – 20 | Percentage of the words in the email that are "85". |
| `word_freq_technology` | float, % | 0 – 7.69 | Percentage of the words in the email that are "technology". |
| `word_freq_1999` | float, % | 0 – 6.89 | Percentage of the words in the email that are "1999". |
| `word_freq_parts` | float, % | 0 – 8.33 | Percentage of the words in the email that are "parts". |
| `word_freq_pm` | float, % | 0 – 11.11 | Percentage of the words in the email that are "pm". |
| `word_freq_direct` | float, % | 0 – 4.76 | Percentage of the words in the email that are "direct". |
| `word_freq_cs` | float, % | 0 – 7.14 | Percentage of the words in the email that are "cs". |
| `word_freq_meeting` | float, % | 0 – 14.28 | Percentage of the words in the email that are "meeting". |
| `word_freq_original` | float, % | 0 – 3.57 | Percentage of the words in the email that are "original". |
| `word_freq_project` | float, % | 0 – 20 | Percentage of the words in the email that are "project". |
| `word_freq_re` | float, % | 0 – 21.42 | Percentage of the words in the email that are "re". |
| `word_freq_edu` | float, % | 0 – 22.05 | Percentage of the words in the email that are "edu". |
| `word_freq_table` | float, % | 0 – 2.17 | Percentage of the words in the email that are "table". |
| `word_freq_conference` | float, % | 0 – 10 | Percentage of the words in the email that are "conference". |
| `char_freq_;` | float, % | 0 – 4.385 | Percentage of the characters in the email that are `;`. |
| `char_freq_(` | float, % | 0 – 9.752 | Percentage of the characters in the email that are `(`. |
| `char_freq_[` | float, % | 0 – 4.081 | Percentage of the characters in the email that are `[`. |
| `char_freq_!` | float, % | 0 – 32.478 | Percentage of the characters in the email that are `!`. |
| `char_freq_$` | float, % | 0 – 6.003 | Percentage of the characters in the email that are `$`. |
| `char_freq_#` | float, % | 0 – 19.829 | Percentage of the characters in the email that are `#`. |
| `capital_run_length_average` | float | 1 – 1102.5 | Average length of the uninterrupted sequences of capital letters. |
| `capital_run_length_longest` | integer | 1 – 9989 | Length of the longest uninterrupted sequence of capital letters. |
| `capital_run_length_total` | integer | 1 – 15841 | Total number of capital letters in the email. |
| `is_spam` | integer (0/1) | 0 – 1 | Target: 1 if the email is spam, 0 if it is legitimate (ham). |
