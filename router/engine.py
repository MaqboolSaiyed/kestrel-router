"""The router a Kestrel employee talks to: a team, how sure it is, and reasons they can check.

Reasons come from three places, none of which needs an API:
  1. The words in the request that pushed the model toward the team.
  2. Policy: when a customer mentions paying but the problem isn't the payment (policy §3).
  3. Precedent: how past requests worded like this one actually ended, and how often the old bot got
     them wrong. This is the reason agents trust most because they can check it against the history.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import data, model

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "models" / "router.joblib"
ASK_BELOW = 0.55      # lowest average cost over three validation quarters; eval/report.md section 4

_PAYMENT = re.compile(r"\b(paid|payment|upi|emi|charged|amount|deducted)\b", re.I)
# Words that carry no routing meaning on their own: greetings, politeness, urgency, payment mentions
# (policy §3), product names (already a field) and ordinary English filler. "not" and "no" stay: "not turning on".
_FILLER = (set("""a about after already am an and any are as at be been but by can do for from get got has have he
help hello hi i if in is it its kindly me my of on or our pls please plz regarding resolve sir so team thank thanks
that the this to urgent very was we what when will with you your namaste good morning evening asap disappointed
call back contact someone order reg paid upi emi payment done extra full want need machine product appliance
air fryer mixer grinder water purifier robot vacuum induction cooktop ceiling fan room heater heaters fans
purifiers fryers mixers vacuums cooktops grinders""".split()))
_BILLING_PROBLEM = re.compile(r"invoice|gst|charged twice|double charge|refund|emi conversion|coupon|deducted", re.I)


class Router:
    def __init__(self, pipeline, neighbours, vectorizer, history, handles, meta):
        self.pipeline, self.neighbours, self.vectorizer = pipeline, neighbours, vectorizer
        self.history, self.handles, self.meta = history, handles, meta
        names = pipeline.named_steps["features"].get_feature_names_out()
        self._word = np.array([n.startswith("text__tfidfvectorizer-1__") for n in names])
        self._names = np.array([n.split("__", 2)[-1] for n in names])

    # ---------------------------------------------------------------- building
    @classmethod
    def train(cls, tr: pd.DataFrame, meta: dict = None) -> "Router":
        pipeline = model.fit(tr, "final_team")
        vec = TfidfVectorizer(preprocessor=model.normalise, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = vec.fit_transform(tr["text"])
        nn = NearestNeighbors(n_neighbors=25, metric="cosine").fit(X)
        history = tr[["final_team", "bot_team"]].reset_index(drop=True)
        teams = pd.read_csv(data._find("teams.csv", data.DATA))
        handles = {data.canonical_team(r.team): r.handles for r in teams.itertuples()}
        return cls(pipeline, nn, vec, history, handles, meta or {})

    def save(self, path: Path = ARTIFACT):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__.copy(), path, compress=3)

    @classmethod
    def load(cls, path: Path = ARTIFACT) -> "Router":
        d = joblib.load(path)
        return cls(d["pipeline"], d["neighbours"], d["vectorizer"], d["history"], d["handles"], d["meta"])

    # ---------------------------------------------------------------- routing
    def route(self, record: dict) -> dict:
        text = data.repair_text(str(record.get("request_text", "")))
        row = pd.DataFrame([{"text": text,
                             "product_family": record.get("product_family") or "unknown",
                             "warranty_status": record.get("warranty_status") or "unknown",
                             "channel": record.get("channel") or "unknown"}])
        pred = model.predict(self.pipeline, row).iloc[0]
        team, conf = pred["team"], float(pred["confidence"])
        ask = conf < ASK_BELOW
        reasons = []

        words = self._signal_words(row, team)
        if ask:
            reasons.append(f"The message doesn't say clearly what's wrong, so the best guess ({team}) is only "
                           f"{conf:.0%} likely. Ask the customer one question before routing.")
        else:
            reasons.append(f"{team} handles: {self.handles.get(team, '')}")
        if words:
            reasons.append("Words that point to " + team + ": " + ", ".join(f'"{w}"' for w in words) + ".")
        if _PAYMENT.search(text) and team != "Billing" and not _BILLING_PROBLEM.search(text):
            reasons.append("Mentions a payment, but the problem described isn't the payment itself. Policy §3: "
                           "that does not make it a Billing request.")
        precedent = self._precedent(text)
        if precedent:
            reasons.append(precedent["sentence"])
        reasons.append(f"Second choice: {pred['second']} ({float(pred['second_confidence']):.0%}).")

        return {"request_id": record.get("request_id"), "team": team, "confidence": round(conf, 3),
                "action": "ask_first" if ask else "route", "second_choice": pred["second"],
                "reasons": reasons, "precedent": {k: v for k, v in (precedent or {}).items() if k != "sentence"},
                "model_version": self.meta.get("version")}

    def _signal_words(self, row: pd.DataFrame, team: str, k: int = 4) -> list:
        x = self.pipeline.named_steps["features"].transform(row)
        clf = self.pipeline.named_steps["clf"]
        coef = model.coefficients(clf)[list(clf.classes_).index(team)]
        x = x.tocsr()
        idx = x.indices[self._word[x.indices]]
        contrib = x[0, idx].toarray().ravel() * coef[idx]
        order = idx[np.argsort(-contrib)]
        picked = []
        for i, c in zip(order, sorted(contrib, reverse=True)):
            if c <= 0 or len(picked) == k:
                break
            w = self._names[i]
            tokens = w.split()
            if tokens[0] in _FILLER or tokens[-1] in _FILLER or all(t in _FILLER | {"not", "no"} for t in tokens):
                continue
            if not any(w in p or p in w for p in picked):
                picked.append(w)
        return picked

    def _precedent(self, text: str, min_similarity: float = 0.5) -> dict:
        dist, idx = self.neighbours.kneighbors(self.vectorizer.transform([text]))
        close = idx[0][dist[0] <= 1 - min_similarity]
        if len(close) < 3:
            return None
        past = self.history.iloc[close]
        top = past["final_team"].value_counts()
        bot_wrong = int((past["bot_team"] != past["final_team"]).sum())
        if top.iloc[0] / len(past) < 0.5:
            sentence = (f"{len(past)} past requests worded like this ended up spread across {len(top)} teams; "
                        f"no team handled more than {top.iloc[0]} of them. The old bot sent {bot_wrong} to the wrong team.")
        else:
            sentence = (f"Of {len(past)} past requests worded like this, {top.iloc[0]} were resolved by {top.index[0]}"
                        + (f"; the old bot sent {bot_wrong} of them to the wrong team." if bot_wrong else "."))
        return {"similar_requests": int(len(past)), "most_common_team": top.index[0],
                "most_common_share": round(float(top.iloc[0] / len(past)), 3), "bot_wrong": bot_wrong,
                "sentence": sentence}
