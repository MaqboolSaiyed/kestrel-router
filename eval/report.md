# Evidence report

Produced by `python -m router evaluate`. Every figure is recomputed from the export.

## 1. What "right" means here

`team_label` is the queue the vendor bot chose when the request came in. `resolution_log.final_team` is the team that actually closed it. They agree on 77.2% of the 10,822 training requests, in every month and in both systems. So a model can match the bot or match where requests really belong, but not both.

How consistent each label is when the exact same text appears more than once (1,950 such requests): the bot's label 98.1%, the team that resolved it 92.1%. The bot is a fixed rule, so it is easy to copy. Where requests really end up depends on things the text doesn't say, which puts a ceiling of roughly 92% on any router that only sees the message.

## 2. Time-based validation

Trained on April 2025 to March 2026 (8,687 requests), tested on April to June 2026 (2,135), the three months just before the period being predicted. All test-period rows come from the new CRM, as do all rows in `test_unlabelled.csv`.

| Router | Agrees with the bot | Sends to the team that resolved it |
|---|---|---|
| The current bot | 100.0% | **76.7%** |
| A model trained to copy the bot's labels | 95.7% | **76.5%** |
| This router (trained on where requests were resolved) | 83.4% | **84.7%** |

A copy of the bot clears the 90% "match" bar easily and inherits all of its mistakes. This router agrees with the bot less, because it is fixing the bot's errors. Its accuracy against real outcomes is 84.7% (95% range 83.2% to 86.3%).

**Stability, and the model choice.** The same experiment on three different quarters. The first version used logistic regression; the calibrated linear SVM was better in every quarter, so it is the one shipped.

| Validation quarter | Trained on | Router (linear SVM) | First version (logistic regression) | Bot |
|---|---|---|---|---|
| Oct 2025 to Dec 2025 | 4,320 | 84.1% | 83.6% | 77.5% |
| Jan 2026 to Mar 2026 | 6,519 | 85.6% | 84.3% | 77.5% |
| Apr 2026 to Jun 2026 | 8,687 | 84.7% | 84.5% | 76.7% |

## 3. By team

| Team | Requests | Router: share caught | Router: share right when it says this team | Bot: share caught |
|---|---|---|---|---|
| Repairs | 498 | 94.2% | 72.5% | 77.3% |
| Installs & Demo | 321 | 82.2% | 88.0% | 69.8% |
| Filters & Consumables | 207 | 82.1% | 93.4% | 83.6% |
| Billing | 255 | 86.3% | 92.4% | 85.9% |
| Returns & Replacement | 326 | 81.0% | 89.8% | 73.0% |
| Warranty Claims | 259 | 81.5% | 85.8% | 73.4% |
| Product Advice | 269 | 78.4% | 92.5% | 77.3% |

## 4. When the router is sure, and when it should ask

| Confidence | Share of requests | Router right | Bot right |
|---|---|---|---|
| 90% to 100% | 53.8% | 97.7% | 88.2% |
| 60% to 90% | 27.4% | 94.0% | 82.9% |
| 40% to 60% | 6.6% | 58.6% | 69.3% |
| 0% to 40% | 12.3% | 21.4% | 16.4% |

Cost of one misroute, from policy §4 and the data: 1.44 transfers on average x Rs 305 + one extra customer contact at Rs 260 = **Rs 698**. Asking the customer one question costs one contact, Rs 260. So it pays to ask whenever the router is less than about 63% likely to be right.

Averaged over the three validation quarters:

| Ask the customer when confidence is below | Asked | Accuracy on the rest | Yearly cost of misroutes + questions |
|---|---|---|---|
| 0% | 0.0% | 84.8% | Rs 9.25 lakh |
| 40% | 12.0% | 93.8% | Rs 6.04 lakh |
| 45% | 13.0% | 94.6% | Rs 5.81 lakh |
| 50% | 14.9% | 95.6% | Rs 5.64 lakh |
| 55% (chosen) | 17.0% | 96.5% | Rs 5.62 lakh |
| 60% | 18.5% | 97.1% | Rs 5.65 lakh |
| 70% | 19.5% | 97.3% | Rs 5.74 lakh |

The lowest average cost is at 55%, and anything from 50% to 60% is within Rs 3,000 a year of it. The router uses 55%. This assumes the customer's answer gets the request to the right team, which is not measured here.

## 5. The cases the service desk complained about

| Case | Requests | Bot right | Router right |
|---|---|---|---|
| Says they paid, but the problem isn't the payment | 128 | 3.1% | 85.9% |
| Water purifier fault | 93 | 24.7% | 97.8% |
| Too vague to route (router under 55% sure) | 369 | 30.4% | 30.6% |

## 6. Would a hosted language model route better?

Tested once on the same validation requests with Groq `openai/gpt-oss-120b`, given each team's description and the Billing rule from policy §3 (`router/llm_experiment.py`).

| Sample | Requests | Hosted model | This router | Bot |
|---|---|---|---|---|
| random | 200 | 85.5% | 86.5% | 76.5% |
| vague (local model under 0.5) | 140 | 12.9% | 23.6% | 22.1% |

It is far better than the bot but no better than the local router, and worse on vague requests. Cost was $0.000026 a request, about $0.02 a month at current volume. Cheap, but it grows with every request and adds an outside dependency for no accuracy gain, so routing stays local. The service only uses it, optionally, to word the question for vague requests.

## 7. What it gets wrong

**Confident and wrong** (27 of 1149 requests above 90%). Mostly records where the closing team contradicts the message, which no router can learn from the text:

- "induction cooktop missing parts in box asap" went to Installs & Demo; router said Returns & Replacement (91%)
- "good morning, difference between lite and pro air fryer" went to Installs & Demo; router said Product Advice (90%)
- "sir box was open, ceiling fan scratched kindly resolve" went to Warranty Claims; router said Returns & Replacement (91%)
- "good morning, received used purifier, want exchange order [number]" went to Product Advice; router said Returns & Replacement (93%)
- "urgent: power consumption of ceiling fan" went to Filters & Consumables; router said Product Advice (91%)
- "hi, cooktop stopped working after 7 months order [number]" went to Billing; router said Repairs (94%)

**Vague** (262 requests under 40%). Requests such as these end up spread across every team, so the router's guess is right 21.4% of the time and the bot's 16.4%:

- "good morning, issue with robot vacuum order [number] pls call back" went to Product Advice
- "pls help - fan problem already paid in full" went to Warranty Claims
- "urgent: someone contact me about mixer grinder" went to Returns & Replacement
- "hello team, service request for machine thanks" went to Repairs
- "hello team, cooktop problem kindly resolve" went to Repairs

## 8. What `predictions.csv` should score

Against the team that resolved each request: about **85%**, most likely between 83.2% and 86.3% (the 95% range on the latest quarter). Three quarters of validation gave 84.1%, 85.6%, 84.7%, and the test period looks the same to the model (16.7% of test requests fall below the ask-first line, against 17.3% in validation).

If it is instead scored against the bot's labels, expect about 83.4%, because it deliberately disagrees with the bot where the bot is usually wrong. Team names are the current ones (Installs & Demo, Filters & Consumables), as the resolution log has recorded them since 15 Jan 2026.

## 9. Rupees

At 8,712 requests a year (current rate) and Rs 698 a misroute, averaged over the three validation quarters:

| Option | Misroutes a year | Cost a year |
|---|---|---|
| Keep the bot | 1,983 | Rs 13.8 lakh + Rs 3.2 lakh licence = **Rs 17.0 lakh** |
| Router, route everything | 1,325 | **Rs 9.2 lakh** |
| Router, ask first when under 55% | 252 + 1,483 questions | **Rs 5.6 lakh** |

Running cost of the router itself: no per-request charge; it runs on an existing machine or a small server. The optional question writer runs only below the ask-first line, about 124 requests a month. Measured on 20 real vague requests: 217 tokens in and 68 out each, $0.00007 a question, so about $0.009 (Rs 0.77) a month.

## 10. Which teams are busiest (for headcount)

Planning on the bot's queues would put people in the wrong teams, because the bot overloads Repairs and Billing and starves Installs & Demo and Returns. Last quarter, per month:

| Team | Bot sent | Actually resolved | Router expects Jul to Sep |
|---|---|---|---|
| Repairs | 206 | 166 | 225 |
| Installs & Demo | 77 | 107 | 102 |
| Filters & Consumables | 95 | 69 | 62 |
| Billing | 110 | 85 | 80 |
| Returns & Replacement | 84 | 109 | 95 |
| Warranty Claims | 66 | 86 | 81 |
| Product Advice | 74 | 90 | 81 |

On top of that, 257 transfers a month land on desks in the current set-up, the work Meenal describes.

## 11. Corrections made to the export

- Read 10,822 training requests, 2,178 test requests, 10,822 resolution records.
- Repaired double-encoded characters in 480 legacy Zoho messages (4.4% of training). The test set has 0.
- Merged the renamed teams onto current names: Installations -> Installs & Demo, Consumables -> Filters & Consumables (policy §5, same responsibilities).
- Moved 4,320 legacy resolution times from UTC to IST. Requests resolved before they were created: 1,140 before the fix, 0 after.
- 126 requests ended in a different team from the bot's with no transfer recorded. Kept for evaluation, flagged so training can leave them out.
- The bot's queue matched the team that resolved the request on 77.2% of training requests.

## 12. Limits

- Scored on the team that closed each request. If Kestrel's own scoring uses the bot's labels, the router will look worse than the bot on paper while sending more customers to the right team.
- The ask-first saving assumes one question gets the request routed correctly. Measure it in the pilot.
- Accuracy will drift as products and wording change. Retraining on each month's closed requests takes a few seconds (`python -m router train`).
- Of the 27 confident errors in validation, every one read for section 7 is a record whose closing team contradicts the message. If that holds for the rest, the true error rate on clean records is lower than the figures above, but the export gives no way to measure that.
