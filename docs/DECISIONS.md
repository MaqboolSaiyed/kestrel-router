# Decision log

Each decision, what else was on the table, and what it costs. Figures are from `python -m router evaluate`.

## A. What to predict

**A1. The target is where a request was resolved, not the bot's label.**
`team_label` is the vendor bot's queue at creation (README, Tanmay's email). `resolution_log.final_team` is the
team that closed the request. They agree on 77.2% of requests, in every month and both systems.

| Option | For | Against | Verdict |
|---|---|---|---|
| Train on `team_label`, as Ritu asked | Meets the stated "90% match" bar (a copy reaches 95.7%) | Copies the bot's errors: 76.5% of requests reach the right team, no better than the bot | rejected |
| **Train on `final_team`** | 84.7% reach the right team; the brief scores against "outcomes you do not have" | Agrees with the bot only 83.4%, so it misses the bar as Ritu worded it | **chosen** |
| Train on `final_team` only where transfers > 0 | Cleaner labels for misroutes | Throws away the 75% of requests the bot got right | rejected |

*Drawback:* Ritu asked for 90% match and gets 83%. The memo says why, up front, with the copy-the-bot result as proof.

**A2. Team names.** Two teams were renamed on 15 Jan 2026 (policy §5, same responsibilities). Training merges old
and new names; predictions use the current names (Installs & Demo, Filters & Consumables), which is how the
resolution log records them for every request since the rename. The whole test period is after it.
*Risk:* if the scorer expects the old names, those two teams would be marked wrong.

**A3. The 126 contradictory records** (closed by a different team from the bot's with no transfer logged) are kept.
Leaving them out of training changed validation accuracy by 0.2 points or less.

## B. Data problems

| Problem | Evidence | Fix |
|---|---|---|
| Legacy Zoho text double-encoded | 480 rows; only three patterns: an ellipsis, a dash, and "Ã©" for the letter e | Replace exactly those patterns, then plain ASCII |
| Legacy resolution times in UTC (policy §9) | 1,140 requests "resolved before created" | +5:30 on legacy rows, 0 left; medians then match (9.5 h both systems) |
| Two team names for each of two teams | Policy §5 | Merged onto current names |
| `first_team` | Identical to `team_label` on every row | Ignored, it adds nothing |
| Exact repeated texts with different outcomes | Resolved team consistent on 92.1% of repeats, bot label on 98.1% | Not fixable; it sets the ceiling, reported as such |
| `sample_submission.csv` | Every row says Repairs | Format reference only |

## C. The model

| Option | Three quarters, share routed right | For | Against | Verdict |
|---|---|---|---|---|
| Logistic regression on TF-IDF word + character n-grams plus product, warranty, channel | 83.6 / 84.3 / 84.5% | Fast, explainable | Slightly behind the SVM in every quarter | first version, replaced |
| **Linear SVM, same features, probabilities calibrated** | **84.1 / 85.6 / 84.7%** | Best in every quarter; still explainable word by word; free to run | Training takes a few seconds longer | **shipped** |
| Complement naive Bayes | 83.1 / 84.8 / 82.8% | Very fast | Less stable | rejected |
| Logistic regression with balanced class weights | 83.8 / 83.7 / 84.0% | Helps small teams in theory | Lower overall | rejected |
| Blend of logistic regression and SVM | 83.9 / 84.8 / 84.8% | | No better than the SVM alone | rejected |
| Hosted language model per request (Groq gpt-oss-120b) | 85.5% on 200 random, 12.9% on 140 vague (router: 86.5%, 23.6%) | No training | No better, worse on vague requests, per-request bill, outside dependency | rejected for routing |
| Product, warranty and channel alone for vague requests | 21.0% on the vague slice (text model 26.5%) | | Worse than the text | rejected |

The ceiling is the labels, not the model: every reasonable setting lands between 84% and 86%.

## D. Ask first instead of guessing

About 12% of requests are too vague for anyone: "please call me about my purifier", "service request for robot
vacuum". Guessing costs a misroute, Rs 698 on average (1.44 transfers x Rs 305 + one extra contact Rs 260); asking
costs one contact, Rs 260. The threshold was picked as the lowest average yearly cost over three quarters: 55%
(Rs 5.62 lakh; anything from 50% to 60% is within Rs 3,000 of it). 17% of requests get a question and 96.5% of the
rest go to the right team. The first version used 60%, chosen for the logistic regression's scores; it was re-tuned
when the model changed because the SVM's confidence is spread differently.
*Assumption, not measured:* the customer's answer gets the request to the right team. The pilot measures it.
`predictions.csv` still gives every request a team, as the brief requires; the ask-first flag is in
`eval/predictions_detail.csv`.

## E. Where the hosted model is used

Only to word the question for vague requests, so it fits what the customer wrote (it answers in Hinglish when the
customer wrote in Hinglish). Optional: without `GROQ_API_KEY`, or on any failure, a fixed question is used and the
response says why. Routing never calls it. Measured on 20 real vague requests: 217 tokens in, 68 out, $0.00007 a
question; about 124 questions a month comes to under Rs 1.

## F. The service

| Choice | Why | Alternative |
|---|---|---|
| FastAPI + one static page | Typed request, automatic rejection of bad input, `/docs` for free, two dependencies | Flask (no validation), Streamlit (heavier, not an endpoint) |
| Retrain on start if the saved model came from another scikit-learn version | A clean machine installs a newer version; retraining takes a few seconds and avoids loading an incompatible file | Pin exact versions (breaks on newer Python) |
| Reasons from three sources: words, policy §3, precedent | An agent can check precedent ("19 of 19 like this were resolved by Repairs; the bot misrouted 6") against history | Generated explanations: unverifiable, need an API |

## G. Data handling

The brief and policy §10 say the client's data must not be published, so the repository is private and shared only
with the reviewers. It still leaves out the CSVs, which the reviewers already have, so the client's raw data exists
in one fewer place. The trained model is included so the service starts straight after cloning; tested on a newer
scikit-learn with no data present, it loads and routes correctly. Customer wording quoted in the evidence report has
order and registration numbers masked. `predictions.csv` holds request IDs and team names only.

## H. Left out

| Left out | Why |
|---|---|
| Live CRM integration | Needs Kestrel's CRM access; the endpoint is the integration point |
| Retraining schedule | One command; Kestrel decides the cadence after the pilot |
| Effort per team from `resolved_at` | Useful for headcount later; the brief asks about volume |
| Multi-issue requests | The data has one final team per request |

## I. Tried and thrown away

1. **Training on the bot's labels.** 95.7% match, 76.5% right. Proof that the requested bar measures the wrong thing.
2. **Logistic regression** as the shipped model (section C).
3. **The 60% ask-first line** once the model changed (section D).
4. **Whole-message re-decoding** for the text damage. It failed on messages mixing damaged and clean characters and
   left fragments like "a€“". Replaced with the three exact patterns.
5. **Raw model words as reasons.** Showed "noise paid", "on", "please". Filtered to words that carry meaning.
6. **Groq for routing** (section C).
7. **Naming the most common past team for vague requests.** For "please call back regarding cooktop" it named one
   team when 25 similar requests were spread across 7. The reason now says they are spread out.
