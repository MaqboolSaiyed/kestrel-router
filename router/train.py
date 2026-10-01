"""Train the router on every labelled request and save it to models/router.joblib."""
from datetime import date
from pathlib import Path

import sklearn

from . import data
from .engine import ARTIFACT, Router


def train_and_save(folder: Path = data.DATA, path: Path = ARTIFACT) -> Router:
    tr, _, _ = data.load(folder)
    meta = {"version": f"{date.today():%Y-%m-%d}", "trained_on": int(len(tr)),
            "target": "team that resolved the request (resolution_log.final_team)",
            "trained_through": f"{tr['created_at'].max():%Y-%m-%d}", "sklearn": sklearn.__version__}
    r = Router.train(tr, meta)
    r.save(path)
    return r
