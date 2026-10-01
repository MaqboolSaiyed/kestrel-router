"""Does a hosted language model route better than the local model? Run once, results kept in eval/.

Same validation requests (Apr to Jun 2026) as the local model, scored against the team that resolved them.
Two samples: a random one, and the vague requests where the local model is under 0.5 confident.

    GROQ_API_KEY=... python -m router.llm_experiment              routing comparison
    GROQ_API_KEY=... python -m router.llm_experiment questions    cost of the optional question writer
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, model

ROOT = Path(__file__).resolve().parent.parent
MODEL = "openai/gpt-oss-120b"
USD_IN, USD_OUT = 0.15, 0.60          # per 1M tokens, console.groq.com/docs/model/openai/gpt-oss-120b
BATCH = 20

SYSTEM = """You route customer service requests for Kestrel Home Appliances to exactly one team.

Teams:
{teams}

Rule from the operations policy: a request belongs to Billing only when the problem is the payment itself \
(invoice, GST, double charge, refund of a payment, EMI conversion). A customer mentioning that they have paid \
does not make it a billing request.

You get a JSON list of requests with id, product, warranty status, channel and text. Return JSON: \
{{"routes": [{{"id": ..., "team": ...}}]}} with one entry per request, team spelled exactly as listed."""


def _teams_text() -> str:
    t = pd.read_csv(data._find("teams.csv", data.DATA))
    return "\n".join(f"- {data.canonical_team(r.team)}: {r.handles}" for r in t.itertuples())


def _call(system: str, items: list, key: str) -> tuple:
    body = {"model": MODEL, "max_completion_tokens": 4000, "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(items, ensure_ascii=False)}]}
    for attempt in range(4):
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "User-Agent": "kestrel-router/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            routes = json.loads(resp["choices"][0]["message"]["content"])["routes"]
            return {str(x["id"]): x["team"] for x in routes}, resp["usage"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("retry-after", 20) or 20) + 1)
                continue
            raise
        except (KeyError, ValueError):
            time.sleep(2)
    return {}, {"prompt_tokens": 0, "completion_tokens": 0}


def main():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("Set GROQ_API_KEY to run this experiment.")
    tr, _, _ = data.load()
    cut = pd.Timestamp("2026-04-01")
    fit_df, val = tr[tr.created_at < cut], tr[tr.created_at >= cut].copy()
    val = val.join(model.predict(model.fit(fit_df, "final_team"), val))
    samples = {"random": val.sample(200, random_state=11),
               "vague (local model under 0.5)": val[val.confidence < 0.5].sample(140, random_state=11)}
    system = SYSTEM.format(teams=_teams_text())
    out, tin, tout = {}, 0, 0
    for name, s in samples.items():
        got = {}
        for i in range(0, len(s), BATCH):
            chunk = s.iloc[i:i + BATCH]
            items = [{"id": r.request_id, "product": r.product_family, "warranty": r.warranty_status,
                      "channel": r.channel, "text": r.text} for r in chunk.itertuples()]
            routes, usage = _call(system, items, key)
            got.update(routes)
            tin += usage.get("prompt_tokens", 0)
            tout += usage.get("completion_tokens", 0)
            print(f"{name}: {min(i + BATCH, len(s))}/{len(s)}", flush=True)
        s = s.assign(llm=s.request_id.map(got))
        answered = s.llm.notna()
        out[name] = {"n": int(len(s)), "answered": int(answered.sum()),
                     "llm_accuracy": float((s.llm[answered] == s.final_team[answered]).mean()),
                     "local_model_accuracy": float((s.team == s.final_team).mean()),
                     "bot_accuracy": float((s.bot_team == s.final_team).mean()),
                     "llm_invalid_team": int((~s.llm[answered].isin(data.TEAMS)).sum()),
                     "answers": {r.request_id: r.llm for r in s.itertuples() if isinstance(r.llm, str)}}
        print(name, out[name], flush=True)
    usd = tin / 1e6 * USD_IN + tout / 1e6 * USD_OUT
    per_request = usd / sum(len(s) for s in samples.values())
    out["cost"] = {"model": MODEL, "tokens_in": tin, "tokens_out": tout, "usd": usd, "usd_per_request": per_request,
                   "requests_per_call": BATCH}
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "llm_experiment.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["cost"], indent=1))


def measure_questions(n: int = 20):
    """Token use of the optional question writer on real vague requests; merged into the same results file."""
    from . import clarify
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY to run this measurement.")
    tr, _, _ = data.load()
    cut = pd.Timestamp("2026-04-01")
    val = tr[tr.created_at >= cut].copy()
    val = val.join(model.predict(model.fit(tr[tr.created_at < cut], "final_team"), val))
    vague = val[val.confidence < 0.55].sample(n, random_state=5)
    tin, tout, ok = [], [], 0
    for r in vague.itertuples():
        out = clarify.ask(r.text, r.product_family, [r.team, r.second], with_usage=True)
        u = out.get("usage")
        if u:
            ok += 1
            tin.append(u.get("prompt_tokens", 0))
            tout.append(u.get("completion_tokens", 0))
        time.sleep(1)
    usd_each = (np.mean(tin) / 1e6 * USD_IN + np.mean(tout) / 1e6 * USD_OUT) if ok else None
    path = ROOT / "eval" / "llm_experiment.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["question_cost"] = {"n": ok, "model": clarify.MODEL, "tokens_in_avg": float(np.mean(tin)) if ok else None,
                            "tokens_out_avg": float(np.mean(tout)) if ok else None, "usd_per_question": usd_each}
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res["question_cost"])


if __name__ == "__main__":
    measure_questions() if sys.argv[1:] == ["questions"] else main()
