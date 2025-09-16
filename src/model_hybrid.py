from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np, pandas as pd

from .embedder import SBertEncoder
from .features import HandcraftedSignals

TEXT_COLS = ["subject","body","cc"]

def _join_cols(df: pd.DataFrame) -> np.ndarray:
    # Join subject + body + CC into a single string per row for TF-IDF
    return (df["subject"].fillna("") + "\n" + df["body"].fillna("") + "\nCC:" + df["cc"].fillna("")).to_numpy()

def build_hybrid(max_features: int = 200_000,
                 ngram_range=(1,2),
                 min_df=2,
                 use_signals: bool = True) -> Pipeline:

    tfidf_branch = Pipeline([
        ("to_text", FunctionTransformer(_join_cols, validate=False)),
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )),
    ])

    sbert_branch = Pipeline([
        ("sbert", SBertEncoder(batch_size=128, show_progress=True)),   # set SBERT_BATCH via env to override
        ("scale", StandardScaler(with_mean=False)),
    ])

    transformers = [
        ("tfidf",  tfidf_branch,  TEXT_COLS),
        ("sbert",  sbert_branch,  TEXT_COLS),
    ]

    if use_signals:
        sig_branch = Pipeline([
            ("sig", HandcraftedSignals()),
            ("scale", StandardScaler(with_mean=False)),
        ])
        transformers.append(("signals", sig_branch, TEXT_COLS))

    pre = ColumnTransformer(transformers, sparse_threshold=0.3)
    clf = LogisticRegression(
        solver="saga",            # handles large sparse + dense
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        n_jobs=-1
    )
    return Pipeline([("pre", pre), ("clf", clf)])
