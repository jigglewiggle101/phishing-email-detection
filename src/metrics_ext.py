from __future__ import annotations
import numpy as np
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             precision_recall_curve)

def summarize_scores(y_true: np.ndarray, scores: np.ndarray, thr: float):
    pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    report = classification_report(y_true, pred, digits=3)
    roc = roc_auc_score(y_true, scores)
    prauc = average_precision_score(y_true, scores)
    return {"threshold": float(thr), "cm": cm.tolist(),
            "roc_auc": float(roc), "pr_auc": float(prauc), "report": report}

def best_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    idx = int(np.nanargmax(f1))
    return float(t[max(0, idx - 1)])  # align t (len-1) with p/r
