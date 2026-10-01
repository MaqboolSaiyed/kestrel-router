"""Load the export and fix what would otherwise mislead the model or the numbers.

Every fix is recorded in a log so the evaluation report can show what was changed.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# Policy §5: two teams were renamed on 15 Jan 2026 with no change in what they handle.
# The test period (Jul to Sep 2026) is entirely after the rename, so the current names are canonical.
RENAMES = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}
TEAMS = ["Repairs", "Installs & Demo", "Filters & Consumables", "Billing", "Returns & Replacement",
         "Warranty Claims", "Product Advice"]

_MOJIBAKE = re.compile("[ÃÂ]|â€")
# The only damage in the export, found by listing every non-ASCII sequence in the legacy rows:
# a triple-encoded ellipsis and dash (punctuation, no meaning), and a two-character sequence standing in
# for the letter e inside words (help, urgent and refund each appear with it).
_JUNK = re.compile("Ã¢â‚¬Â¦"          # ellipsis
                   "|Ã¢â‚¬â€œ"    # dash
                   "|â€¦|â€“|â€”")
_E_ACUTE = "Ã©"


def _find(name: str, folder: Path) -> Path:
    for p in folder.iterdir():
        if p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"{name} not found in {folder}")


def repair_text(s: str) -> str:
    """Undo the legacy Zoho encoding damage, then reduce to plain ASCII."""
    s = _JUNK.sub(" ", s).replace(_E_ACUTE, "e")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c if ord(c) < 128 else " " for c in s)   # any stray dash, quote or ellipsis


def canonical_team(name: str) -> str:
    return RENAMES.get(name, name)


def load(folder: Path = DATA) -> tuple:
    """Returns (train, test, log). train carries bot_team (what the bot chose) and final_team (where it was
    actually resolved), both in current team names."""
    log = []
    tr = pd.read_csv(_find("train.csv", folder), dtype=str, keep_default_na=False)
    te = pd.read_csv(_find("test_unlabelled.csv", folder), dtype=str, keep_default_na=False)
    rl = pd.read_csv(_find("resolution_log.csv", folder), dtype=str, keep_default_na=False)
    log.append(f"Read {len(tr):,} training requests, {len(te):,} test requests, {len(rl):,} resolution records.")

    for df in (tr, te):
        df["text"] = df["request_text"].map(repair_text)
        df["created_at"] = pd.to_datetime(df["created_at_ist"])
    damaged = tr["request_text"].str.contains(_MOJIBAKE)
    log.append(f"Repaired double-encoded characters in {int(damaged.sum()):,} legacy Zoho messages "
               f"({damaged.mean():.1%} of training). The test set has "
               f"{int(te['request_text'].str.contains(_MOJIBAKE).sum())}.")

    tr = tr.merge(rl, on="request_id", how="left", validate="one_to_one")
    tr["bot_team"] = tr["team_label"].map(canonical_team)
    tr["final_team"] = tr["final_team"].map(canonical_team)
    tr["first_team"] = tr["first_team"].map(canonical_team)
    log.append("Merged the renamed teams onto current names: Installations -> Installs & Demo, "
               "Consumables -> Filters & Consumables (policy §5, same responsibilities).")

    # Policy §9: Zoho resolution events were stored in UTC and never converted.
    tr["resolved_at"] = pd.to_datetime(tr["resolved_at"])
    legacy = tr["source"] == "legacy_zoho"
    neg_before = int(((tr["resolved_at"] < tr["created_at"]) & legacy).sum())
    tr.loc[legacy, "resolved_at"] += pd.Timedelta(hours=5, minutes=30)
    neg_after = int((tr["resolved_at"] < tr["created_at"]).sum())
    log.append(f"Moved {int(legacy.sum()):,} legacy resolution times from UTC to IST. Requests resolved before "
               f"they were created: {neg_before:,} before the fix, {neg_after} after.")
    tr["hours_to_resolve"] = (tr["resolved_at"] - tr["created_at"]).dt.total_seconds() / 3600
    tr["transfers"] = pd.to_numeric(tr["transfers"], errors="coerce").fillna(0).astype(int)
    tr["bot_right"] = tr["bot_team"] == tr["final_team"]

    # The resolution log contradicts itself on a small set: the request ended in a different team from
    # the one the bot chose, yet no transfer was recorded. Their final team can't be trusted as a label.
    tr["label_suspect"] = ~tr["bot_right"] & (tr["transfers"] == 0)
    log.append(f"{int(tr['label_suspect'].sum())} requests ended in a different team from the bot's with no transfer "
               f"recorded. Kept for evaluation, flagged so training can leave them out.")
    log.append(f"The bot's queue matched the team that resolved the request on {tr['bot_right'].mean():.1%} of "
               f"training requests.")
    return tr, te, log
