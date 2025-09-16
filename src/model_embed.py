from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .embedder import SBertEncoder
from .features import HandcraftedSignals

TEXT_COLS = ["subject","body","cc"]

def build_embed_only(use_signals: bool = True) -> Pipeline:
    # Embedding branch
    sbert_branch = Pipeline([
        ("sbert", SBertEncoder()),
        ("scale", StandardScaler(with_mean=False)),
    ])

    transformers = [("sbert", sbert_branch, TEXT_COLS)]

    if use_signals:
        signal_branch = Pipeline([
            ("sig", HandcraftedSignals()),
            ("scale", StandardScaler(with_mean=False)),
        ])
        transformers.append(("signals", signal_branch, TEXT_COLS))

    pre = ColumnTransformer(transformers, sparse_threshold=0.3)
    clf = LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", max_iter=400)
    return Pipeline([("pre", pre), ("clf", clf)])
