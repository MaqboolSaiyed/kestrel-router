"""Command line.

  python -m router train       train on data/ and save models/router.joblib
  python -m router predict     write predictions.csv for data/test_unlabelled.csv
  python -m router evaluate    validation report into eval/report.md
  python -m router serve       start the service and screen on http://127.0.0.1:8000
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cmd_train(a):
    from . import train
    r = train.train_and_save()
    print(f"Trained on {r.meta['trained_on']:,} requests, saved models/router.joblib")


def cmd_predict(a):
    import pandas as pd
    from . import data, model
    from .engine import ARTIFACT, ASK_BELOW, Router
    try:
        r = Router.load(ARTIFACT)
    except Exception:
        from . import train
        r = train.train_and_save()
    _, te, _ = data.load()
    p = model.predict(r.pipeline, te)
    out = pd.DataFrame({"request_id": te["request_id"], "team": p["team"].values})
    out.to_csv(ROOT / a.out, index=False)
    detail = out.assign(confidence=p["confidence"].round(3).values, second_choice=p["second"].values,
                        action=["ask_first" if c < ASK_BELOW else "route" for c in p["confidence"]])
    detail.to_csv(ROOT / "eval" / "predictions_detail.csv", index=False)
    ask = (p["confidence"] < ASK_BELOW).mean()
    print(f"Wrote {len(out):,} rows to {a.out}. {ask:.1%} are below the ask-first line "
          f"(detail in eval/predictions_detail.csv).")
    print(out["team"].value_counts().to_string())


def cmd_evaluate(a):
    from . import evaluate
    evaluate.run()


def cmd_serve(a):
    import uvicorn
    uvicorn.run("router.service:app", host=a.host, port=a.port)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="router", description="Kestrel service-request router")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    pr = sub.add_parser("predict")
    pr.add_argument("--out", default="predictions.csv")
    pr.set_defaults(fn=cmd_predict)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
