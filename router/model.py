"""The routing model: request text plus the three structured fields, into one of the seven teams.

Character n-grams cope with spelling slips and run-together words; word n-grams carry phrases like
"stopped working" or "reschedule installation"; product, warranty status and channel are one-hot.
A linear SVM with calibrated probabilities beat logistic regression in each of three validation
quarters (84.8% against 84.2% on average); both are linear, so every prediction can still be traced
to the words that pushed it.
"""
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_union
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

META = ["product_family", "warranty_status", "channel"]
_IDS = re.compile(r"\b(?:ko|sr)\d+\b|\b(?:order|reg)\s+no\b", re.I)


def normalise(text: str) -> str:
    """Order and registration numbers are unique per customer, so they only add noise."""
    return _IDS.sub(" ", text.lower())


def build(kind: str = "svm", C: float = None, meta: bool = True) -> Pipeline:
    text = make_union(
        TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
    )
    parts = [("text", text, "text")]
    if meta:
        parts.append(("meta", OneHotEncoder(handle_unknown="ignore"), META))
    if kind == "svm":
        clf = CalibratedClassifierCV(LinearSVC(C=C or 0.1), cv=5)
    else:
        clf = LogisticRegression(C=C or 4.0, max_iter=3000)
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", clf)])


def fit(df: pd.DataFrame, target: str, **kw) -> Pipeline:
    return build(**kw).fit(df[["text"] + META], df[target])


def coefficients(clf) -> np.ndarray:
    """Per-class word weights. For the calibrated SVM, the average over its internal folds."""
    if hasattr(clf, "coef_"):
        return clf.coef_
    fitted = [getattr(c, "estimator", None) or getattr(c, "base_estimator") for c in clf.calibrated_classifiers_]
    return np.mean([f.coef_ for f in fitted], axis=0)


def predict(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(df[["text"] + META])
    classes = model.classes_
    order = proba.argsort(axis=1)[:, ::-1]
    return pd.DataFrame({
        "team": classes[order[:, 0]], "confidence": proba.max(axis=1),
        "second": classes[order[:, 1]], "second_confidence": proba[range(len(df)), order[:, 1]],
    }, index=df.index)
