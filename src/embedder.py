# from __future__ import annotations
# import numpy as np, pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sentence_transformers import SentenceTransformer

# class SBertEncoder(BaseEstimator, TransformerMixin):
#     """
#     Encodes subject+body(+cc) into a single MiniLM embedding per email.
#     Expects a DataFrame with columns ['subject','body','cc'].
#     """
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
#         self.model_name = model_name
#         self.batch_size = batch_size
#         self.model_ = None
#         self.dim_ = None

#     def fit(self, X: pd.DataFrame, y=None):
#         self.model_ = SentenceTransformer(self.model_name)
#         sample = self._join(X.iloc[0]) if len(X) else ""
#         v = self.model_.encode([sample], convert_to_numpy=True)
#         self.dim_ = int(v.shape[1])
#         return self

#     def transform(self, X: pd.DataFrame) -> np.ndarray:
#         assert self.model_ is not None, "Call fit() first"
#         texts = X.apply(self._join, axis=1).tolist()
#         emb = self.model_.encode(
#             texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True
#         )
#         return emb

#     @staticmethod
#     def _join(r: pd.Series) -> str:
#         return f"{r.get('subject','')}\n{r.get('body','')}\nCC:{r.get('cc','')}"

# src/embedder.py
from __future__ import annotations
import os, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SBertEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64, show_progress: bool = True,
                 device: str | None = None, max_seq_len: int = 128):
        self.model_name = model_name
        self.batch_size = int(os.getenv("SBERT_BATCH", batch_size))
        self.show_progress = show_progress
        self.device = device
        self.max_seq_len = max_seq_len
        self.model_ = None
        self.dim_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.model_ = SentenceTransformer(self.model_name, device=self.device)
        self.model_.max_seq_length = self.max_seq_len
        # poke once to get dim
        sample = self._join(X.iloc[0]) if len(X) else ""
        v = self.model_.encode([sample], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        self.dim_ = int(v.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "Call fit() first"
        texts = X.apply(self._join, axis=1).tolist()
        emb = self.model_.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
        )
        return emb

    @staticmethod
    def _join(r: pd.Series) -> str:
        return f"{r.get('subject','')}\n{r.get('body','')}\nCC:{r.get('cc','')}"
