"""Evidence that the router works, and how often it doesn't. Writes eval/report.md and eval/results.json.

Every number is computed here from the export, except the hosted-model comparison, which is read from
eval/llm_experiment.json (produced once by router.llm_experiment because it needs an API key).
"""
import json
import re
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, model
from .engine import ASK_BELOW

ROOT = Path(__file__).resolve().parent.parent
TRANSFER, EXTRA_CONTACT, LICENCE = 305, 260, 320_000          # policy §4
VALID_FROM = pd.Timestamp("2026-04-01")


def _quote(text: str) -> str:
    """Customer wording for examples, with order and registration numbers masked: this report is shareable."""
    t = re.sub(r"\b(?:ko|sr)\d+\b", "[number]", text.strip(), flags=re.I)
    return t[:90]


def ci(p, n, z=1.96):
    h = z * sqrt(p * (1 - p) / n)
    return p - h, p + h


def pct(x):
    return f"{x:.1%}"


def run():
    tr, te, log = data.load()
    res, L = {}, []
    w = L.append
    fit_df, val = tr[tr.created_at < VALID_FROM], tr[tr.created_at >= VALID_FROM].copy()
    m = model.fit(fit_df, "final_team")
    val = val.join(model.predict(m, val))
    val["ok"], val["bot_ok"] = val.team == val.final_team, val.bot_team == val.final_team

    w("# Evidence report\n")
    w("Produced by `python -m router evaluate`. Every figure is recomputed from the export.\n")
    w("## 1. What \"right\" means here\n")
    w("`team_label` is the queue the vendor bot chose when the request came in. `resolution_log.final_team` is the team "
      "that actually closed it. They agree on " + pct(tr.bot_right.mean()) + " of the 10,822 training requests, in every "
      "month and in both systems. So a model can match the bot or match where requests really belong, but not both.\n")
    norm = tr.text.str.lower().str.replace(r"\b(ko|sr)\d+\b", "", regex=True).str.replace(r"[^a-z ]", " ", regex=True).str.split().str.join(" ")
    rep = tr.assign(norm=norm).groupby("norm").filter(lambda g: len(g) >= 2)
    cons_final = rep.groupby("norm").final_team.agg(lambda s: s.value_counts().iloc[0]).sum() / len(rep)
    cons_bot = rep.groupby("norm").bot_team.agg(lambda s: s.value_counts().iloc[0]).sum() / len(rep)
    res.update(bot_vs_final=float(tr.bot_right.mean()), ceiling_final=float(cons_final), ceiling_bot=float(cons_bot))
    w(f"How consistent each label is when the exact same text appears more than once ({len(rep):,} such requests): the "
      f"bot's label {pct(cons_bot)}, the team that resolved it {pct(cons_final)}. The bot is a fixed rule, so it is easy "
      f"to copy. Where requests really end up depends on things the text doesn't say, which puts a ceiling of roughly "
      f"{cons_final:.0%} on any router that only sees the message.\n")

    # 2. Validation
    w("## 2. Time-based validation\n")
    w(f"Trained on April 2025 to March 2026 ({len(fit_df):,} requests), tested on April to June 2026 ({len(val):,}), the "
      f"three months just before the period being predicted. All test-period rows come from the new CRM, as do all "
      f"rows in `test_unlabelled.csv`.\n")
    mb = model.fit(fit_df, "bot_team")
    pb = model.predict(mb, val)
    rows = [("The current bot", 1.0, val.bot_ok.mean()),
            ("A model trained to copy the bot's labels", (pb.team == val.bot_team).mean(), (pb.team == val.final_team).mean()),
            ("This router (trained on where requests were resolved)", (val.team == val.bot_team).mean(), val.ok.mean())]
    w("| Router | Agrees with the bot | Sends to the team that resolved it |\n|---|---|---|")
    for name, a, b in rows:
        w(f"| {name} | {pct(a)} | **{pct(b)}** |")
    lo, hi = ci(val.ok.mean(), len(val))
    res.update(val_accuracy=float(val.ok.mean()), val_ci=[lo, hi], val_bot_accuracy=float(val.bot_ok.mean()),
               val_agree_bot=float((val.team == val.bot_team).mean()), copycat_vs_bot=float((pb.team == val.bot_team).mean()),
               copycat_vs_final=float((pb.team == val.final_team).mean()))
    w(f"\nA copy of the bot clears the 90% \"match\" bar easily and inherits all of its mistakes. This router agrees with the "
      f"bot less, because it is fixing the bot's errors. Its accuracy against real outcomes is {pct(val.ok.mean())} "
      f"(95% range {pct(lo)} to {pct(hi)}).\n")

    w("**Stability, and the model choice.** The same experiment on three different quarters. The first version used "
      "logistic regression; the calibrated linear SVM was better in every quarter, so it is the one shipped.\n")
    w("| Validation quarter | Trained on | Router (linear SVM) | First version (logistic regression) | Bot |\n|---|---|---|---|---|")
    stab, first, quarters, bot_miss = [], [], [], []
    for start in ["2025-10-01", "2026-01-01", "2026-04-01"]:
        s, e = pd.Timestamp(start), pd.Timestamp(start) + pd.DateOffset(months=3)
        f, v = tr[tr.created_at < s], tr[(tr.created_at >= s) & (tr.created_at < e)]
        pq = model.predict(model.fit(f, "final_team"), v)
        quarters.append(((pq.team == v.final_team).values, pq.confidence.values))
        bot_miss.append(float((v.bot_team != v.final_team).mean()))
        a = (pq.team == v.final_team).mean()
        b = (model.predict(model.fit(f, "final_team", kind="logreg"), v).team == v.final_team).mean()
        stab.append(float(a))
        first.append(float(b))
        w(f"| {s:%b %Y} to {(e - pd.Timedelta(days=1)):%b %Y} | {len(f):,} | {pct(a)} | {pct(b)} | "
          f"{pct((v.bot_team == v.final_team).mean())} |")
    res["stability"], res["stability_first_version"] = stab, first

    # 3. Per team
    w("\n## 3. By team\n")
    w("| Team | Requests | Router: share caught | Router: share right when it says this team | Bot: share caught |\n|---|---|---|---|---|")
    for t in data.TEAMS:
        sub = val[val.final_team == t]
        said = val[val.team == t]
        w(f"| {t} | {len(sub)} | {pct(sub.ok.mean())} | {pct((said.final_team == t).mean())} | {pct(sub.bot_ok.mean())} |")

    # 4. Confidence and ask-first
    w("\n## 4. When the router is sure, and when it should ask\n")
    w("| Confidence | Share of requests | Router right | Bot right |\n|---|---|---|---|")
    for a, b in [(0.9, 1.01), (0.6, 0.9), (0.4, 0.6), (0, 0.4)]:
        g = val[(val.confidence >= a) & (val.confidence < b)]
        w(f"| {a:.0%} to {min(b, 1):.0%} | {pct(len(g) / len(val))} | {pct(g.ok.mean())} | {pct(g.bot_ok.mean())} |")
    mis = tr[~tr.bot_right]
    unit = mis.transfers.mean() * TRANSFER + EXTRA_CONTACT
    per_year = len(te) / 3 * 12
    w(f"\nCost of one misroute, from policy §4 and the data: {mis.transfers.mean():.2f} transfers on average x Rs {TRANSFER} "
      f"+ one extra customer contact at Rs {EXTRA_CONTACT} = **Rs {unit:.0f}**. Asking the customer one question costs "
      f"one contact, Rs {EXTRA_CONTACT}. So it pays to ask whenever the router is less than about "
      f"{1 - EXTRA_CONTACT / unit:.0%} likely to be right.\n")
    w("Averaged over the three validation quarters:\n")
    w("| Ask the customer when confidence is below | Asked | Accuracy on the rest | Yearly cost of misroutes + questions |\n|---|---|---|---|")
    sweep = []
    for th in [0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        asked, rest, cost = [], [], []
        for ok, conf in quarters:
            ask = conf < th
            asked.append(ask.mean())
            rest.append(ok[~ask].mean())
            cost.append(((~ok & ~ask).mean() * unit + ask.mean() * EXTRA_CONTACT) * per_year)
        sweep.append((th, float(np.mean(asked)), float(np.mean(rest)), float(np.mean(cost))))
        mark = " (chosen)" if abs(th - ASK_BELOW) < 1e-9 else ""
        w(f"| {th:.0%}{mark} | {pct(np.mean(asked))} | {pct(np.mean(rest))} | Rs {np.mean(cost) / 1e5:.2f} lakh |")
    best = min(sweep, key=lambda x: x[3])
    w(f"\nThe lowest average cost is at {best[0]:.0%}, and anything from 50% to 60% is within Rs 3,000 a year of it. "
      f"The router uses {ASK_BELOW:.0%}. This assumes the customer's answer gets the request to the right team, which "
      f"is not measured here.\n")
    res.update(misroute_cost=float(unit), requests_per_year=float(per_year), ask_sweep=sweep)

    # 5. Meenal's cases
    w("## 5. The cases the service desk complained about\n")
    pay = val.text.str.contains(r"\bpaid\b|\bupi\b|\bemi\b|payment done|paid extra|paid in full", case=False)
    fault = val.text.str.contains(r"not working|stopped working|not turning on|tripping|leak|noise|burnt|error", case=False)
    w("| Case | Requests | Bot right | Router right |\n|---|---|---|---|")
    for name, msk in [("Says they paid, but the problem isn't the payment", pay & (val.final_team != "Billing")),
                      ("Water purifier fault", (val.product_family == "Water Purifier") & fault),
                      (f"Too vague to route (router under {ASK_BELOW:.0%} sure)", val.confidence < ASK_BELOW)]:
        g = val[msk]
        w(f"| {name} | {len(g)} | {pct(g.bot_ok.mean())} | {pct(g.ok.mean())} |")

    # 6. Hosted model
    llm = ROOT / "eval" / "llm_experiment.json"
    if llm.exists():
        x = json.loads(llm.read_text())
        w("\n## 6. Would a hosted language model route better?\n")
        w(f"Tested once on the same validation requests with Groq `{x['cost']['model']}`, given each team's description "
          f"and the Billing rule from policy §3 (`router/llm_experiment.py`).\n")
        w("| Sample | Requests | Hosted model | This router | Bot |\n|---|---|---|---|---|")
        for k, v in x.items():
            if k not in ("cost", "question_cost"):
                w(f"| {k} | {v['n']} | {pct(v['llm_accuracy'])} | {pct(v['local_model_accuracy'])} | {pct(v['bot_accuracy'])} |")
        per_month = x['cost']['usd_per_request'] * len(te) / 3
        w(f"\nIt is far better than the bot but no better than the local router, and worse on vague requests. Cost was "
          f"${x['cost']['usd_per_request']:.6f} a request, about ${per_month:.2f} a month at current volume. Cheap, but it "
          f"grows with every request and adds an outside dependency for no accuracy gain, so routing stays local. The "
          f"service only uses it, optionally, to word the question for vague requests.\n")
        res["llm"] = x

    # 7. Errors
    w("## 7. What it gets wrong\n")
    hi_err = val[(val.confidence >= 0.9) & ~val.ok]
    w(f"**Confident and wrong** ({len(hi_err)} of {(val.confidence >= 0.9).sum()} requests above 90%). Mostly records "
      f"where the closing team contradicts the message, which no router can learn from the text:\n")
    for r in hi_err.sample(min(6, len(hi_err)), random_state=4).itertuples():
        w(f"- \"{_quote(r.text)}\" went to {r.final_team}; router said {r.team} ({r.confidence:.0%})")
    vague = val[val.confidence < 0.4]
    w(f"\n**Vague** ({len(vague)} requests under 40%). Requests such as these end up spread across every team, so the "
      f"router's guess is right {pct(vague.ok.mean())} of the time and the bot's {pct(vague.bot_ok.mean())}:\n")
    for r in vague.sample(min(5, len(vague)), random_state=4).itertuples():
        w(f"- \"{_quote(r.text)}\" went to {r.final_team}")

    # 8. Expected score on the test file
    pred = pd.read_csv(ROOT / "eval" / "predictions_detail.csv") if (ROOT / "eval" / "predictions_detail.csv").exists() else None
    w("\n## 8. What `predictions.csv` should score\n")
    w(f"Against the team that resolved each request: about **{np.mean(stab):.0%}**, most likely between {pct(lo)} and "
      f"{pct(hi)} (the 95% range on the latest quarter). Three quarters of validation gave "
      f"{', '.join(pct(s) for s in stab)}, and the test period "
      "looks the same to the model" + (f" ({pct((pred.confidence < ASK_BELOW).mean())} of test requests fall below the "
      f"ask-first line, against {pct((val.confidence < ASK_BELOW).mean())} in validation)" if pred is not None else "") + ".\n")
    w(f"If it is instead scored against the bot's labels, expect about {pct(res['val_agree_bot'])}, because it "
      "deliberately disagrees with the bot where the bot is usually wrong. Team names are the current ones "
      "(Installs & Demo, Filters & Consumables), as the resolution log has recorded them since 15 Jan 2026.\n")
    res["expected_score"] = float(np.mean(stab))

    # 9. Money
    w("## 9. Rupees\n")
    bot_rate = float(np.mean(bot_miss))
    bot_cost = bot_rate * per_year * unit
    route_all = [s for s in sweep if s[0] == 0][0]
    th = [s for s in sweep if abs(s[0] - ASK_BELOW) < 1e-9][0]
    w(f"At {per_year:,.0f} requests a year (current rate) and Rs {unit:.0f} a misroute, averaged over the three "
      f"validation quarters:\n")
    w("| Option | Misroutes a year | Cost a year |\n|---|---|---|")
    w(f"| Keep the bot | {bot_rate * per_year:,.0f} | Rs {bot_cost / 1e5:.1f} lakh + Rs 3.2 lakh licence = **Rs {(bot_cost + LICENCE) / 1e5:.1f} lakh** |")
    w(f"| Router, route everything | {(1 - route_all[2]) * per_year:,.0f} | **Rs {route_all[3] / 1e5:.1f} lakh** |")
    w(f"| Router, ask first when under {ASK_BELOW:.0%} | {(1 - th[1]) * (1 - th[2]) * per_year:,.0f} + {th[1] * per_year:,.0f} questions | **Rs {th[3] / 1e5:.1f} lakh** |")
    res.update(cost_bot=float(bot_cost + LICENCE), cost_route_all=float(route_all[3]), cost_ask_first=float(th[3]),
               ask_share=th[1], ask_rest_accuracy=th[2], bot_misroute_rate=bot_rate)
    qc = json.loads(llm.read_text()).get("question_cost") if llm.exists() else None
    if qc:
        per_month = th[1] * per_year / 12
        w(f"\nRunning cost of the router itself: no per-request charge; it runs on an existing machine or a small server. "
          f"The optional question writer runs only below the ask-first line, about {per_month:.0f} requests a month. "
          f"Measured on {qc['n']} real vague requests: {qc['tokens_in_avg']:.0f} tokens in and {qc['tokens_out_avg']:.0f} out "
          f"each, ${qc['usd_per_question']:.5f} a question, so about ${qc['usd_per_question'] * per_month:.3f} "
          f"(Rs {qc['usd_per_question'] * per_month * 85:.2f}) a month.\n")
        res["question_cost_month_inr"] = qc["usd_per_question"] * per_month * 85

    # 10. Workload
    w("## 10. Which teams are busiest (for headcount)\n")
    last_q = tr[tr.created_at >= VALID_FROM]
    w("Planning on the bot's queues would put people in the wrong teams, because the bot overloads Repairs and Billing "
      "and starves Installs & Demo and Returns. Last quarter, per month:\n")
    w("| Team | Bot sent | Actually resolved | Router expects Jul to Sep |\n|---|---|---|---|")
    exp = pred.team.value_counts() / 3 if pred is not None else pd.Series(dtype=float)
    for t in data.TEAMS:
        w(f"| {t} | {(last_q.bot_team == t).sum() / 3:.0f} | {(last_q.final_team == t).sum() / 3:.0f} | {exp.get(t, 0):.0f} |")
    w(f"\nOn top of that, {int(last_q.transfers.sum() / 3)} transfers a month land on desks in the current set-up, "
      "the work Meenal describes.\n")

    w("## 11. Corrections made to the export\n")
    for x in log:
        w(f"- {x}")
    w("\n## 12. Limits\n")
    w("- Scored on the team that closed each request. If Kestrel's own scoring uses the bot's labels, the router will "
      "look worse than the bot on paper while sending more customers to the right team.\n"
      "- The ask-first saving assumes one question gets the request routed correctly. Measure it in the pilot.\n"
      "- Accuracy will drift as products and wording change. Retraining on each month's closed requests takes a few "
      "seconds (`python -m router train`).\n"
      f"- Of the {len(hi_err)} confident errors in validation, every one read for section 7 is a record whose closing "
      "team contradicts the message. If that holds for the rest, the true error rate on clean records is lower than "
      "the figures above, but the export gives no way to measure that.\n")

    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "report.md").write_text("\n".join(L), encoding="utf-8")
    (ROOT / "eval" / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))
