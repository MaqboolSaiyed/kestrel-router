"""Checks that run without an API key: python tests/test_router.py  (pytest also works)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.pop("GROQ_API_KEY", None)

import pandas as pd  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from router import clarify, data, service  # noqa: E402

service.startup()
ROUTER = service.router

CASES = [  # (text, product, expected team or "ask_first")
    ("sir mixer grinder making loud noise paid on upi pls call back", "Mixer Grinder", "Repairs"),
    ("hi, water purifier not turning on since yesterday", "Water Purifier", "Repairs"),
    ("charged twice for my air fryer order, need refund of the extra amount", "Air Fryer", "Billing"),
    ("want to reschedule installation of water purifier i paid extra for this", "Water Purifier", "Installs & Demo"),
    ("need new filter candle for my purifier", "Water Purifier", "Filters & Consumables"),
    ("box was open and the fan blade is broken, want replacement", "Ceiling Fan", "Returns & Replacement"),
    ("how do i register my kestrel shield plan", "Robot Vacuum", "Warranty Claims"),
    ("can the air fryer run on an inverter", "Air Fryer", "Product Advice"),
    ("please call back regarding my cooktop", "Induction Cooktop", "ask_first"),
]


def _route(text, product=None):
    return service.route(service.ServiceRequest(request_text=text, product_family=product))


def test_service_starts_without_key():
    h = service.health()
    assert h["ready"] and h["error"] is None
    assert "standard question" in h["question_writer"]


def test_known_cases():
    wrong = []
    for text, product, want in CASES:
        out = _route(text, product)
        got = "ask_first" if out["action"] == "ask_first" else out["team"]
        if got != want:
            wrong.append((text, want, got, out["confidence"]))
    assert not wrong, wrong


def test_response_shape():
    out = _route("water purifier leaking from the bottom", "Water Purifier")
    assert out["team"] in data.TEAMS and out["second_choice"] in data.TEAMS
    assert 0 <= out["confidence"] <= 1 and out["action"] in ("route", "ask_first")
    assert out["reasons"] and all(isinstance(r, str) and r for r in out["reasons"])


def test_vague_request_gets_a_question_without_key():
    out = _route("please call back regarding my cooktop", "Induction Cooktop")
    assert out["action"] == "ask_first"
    assert out["ask_customer"]["question"] and "no GROQ_API_KEY" in out["ask_customer"]["source"]


def test_bad_key_fails_politely():
    os.environ["GROQ_API_KEY"] = "invalid-test-key"
    try:
        q = clarify.ask("please call back", "Room Heater", ["Repairs", "Billing"])
    finally:
        os.environ.pop("GROQ_API_KEY", None)
    assert q["question"] and q["source"].startswith("standard question (")


def test_rejects_bad_input():
    for bad in [{"request_text": ""}, {"request_text": "hi", "product_family": "Toaster"},
                {"request_text": "hi", "channel": "fax"}, {}]:
        try:
            service.ServiceRequest(**bad)
        except ValidationError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_text_repair():
    e = "Ã©"
    assert data.repair_text(f"h{e}lp urg{e}nt") == "help urgent"
    assert data.repair_text("very disappointed Ã¢â‚¬Â¦").strip() == "very disappointed"
    assert data.repair_text("plain text") == "plain text"


def test_predictions_file():
    p = pd.read_csv(ROOT / "predictions.csv")
    s = pd.read_csv(data._find("sample_submission.csv", data.DATA)) if (data.DATA / "sample_submission.csv").exists() else None
    assert list(p.columns) == ["request_id", "team"] and p.request_id.is_unique
    assert p.team.isin(data.TEAMS).all()
    if s is not None:
        assert (p.request_id.values == s.request_id.values).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
