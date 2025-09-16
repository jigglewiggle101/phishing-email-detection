from __future__ import annotations
import os, re, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SUSPICIOUS_PHRASES = [
    "urgent action required","verify your account","validate your account",
    "password expires","confirm immediately","update your details",
    "click the link","unusual sign in","account on hold","reset now",
]
URL_RE   = re.compile(r"https?://[^\s]+", re.I)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+", re.I)
ORG_DOMAIN = os.getenv("ORG_DOMAIN", "")

class HandcraftedSignals(BaseEstimator, TransformerMixin):
    """Numeric side-features from subject/body/cc for extra signal + explainability."""
    def __init__(self):
        self.feature_names_ = [
            "num_urls","num_emails","exclaim_count","all_caps_ratio",
            "suspicious_phrase_hits","cc_count","cc_unique_domains","cc_external_ratio",
        ]

    def fit(self, X, y=None): return self

    def transform(self, X: pd.DataFrame):
        subj = X.get("subject", pd.Series(dtype=str)).fillna("")
        body = X.get("body", pd.Series(dtype=str)).fillna("")
        cc   = X.get("cc", pd.Series(dtype=str)).fillna("")
        joined = subj + "\n" + body

        def caps_ratio(t: str) -> float:
            letters = [c for c in t if c.isalpha()]
            return (sum(1 for c in letters if c.isupper()) / max(1, len(letters))) if letters else 0.0

        num_urls   = joined.apply(lambda t: len(URL_RE.findall(t))).to_numpy()
        num_emails = joined.apply(lambda t: len(EMAIL_RE.findall(t))).to_numpy()
        exclaims   = joined.apply(lambda t: t.count("!")).to_numpy()
        caps       = joined.apply(caps_ratio).to_numpy()
        low        = joined.str.lower()
        sus_hits   = low.apply(lambda t: sum(p in t for p in SUSPICIOUS_PHRASES)).to_numpy()

        # CC-derived
        def split_emails(s: str): return [e.strip() for e in EMAIL_RE.findall(s)]
        cc_lists   = cc.apply(split_emails)
        cc_count   = cc_lists.apply(len).to_numpy()
        def dom(e: str): return e.split("@")[-1].lower() if "@" in e else ""
        cc_domains = cc_lists.apply(lambda L: [dom(e) for e in L])
        cc_unique  = cc_domains.apply(lambda L: len(set(d for d in L if d))).to_numpy()
        if ORG_DOMAIN:
            cc_ext = cc_domains.apply(lambda L: (sum(1 for d in L if d and d != ORG_DOMAIN) / max(1, len(L))) if L else 0.0).to_numpy()
        else:
            cc_ext = np.zeros_like(cc_count, dtype=float)

        return np.vstack([num_urls,num_emails,exclaims,caps,sus_hits,cc_count,cc_unique,cc_ext]).T.astype(float)
