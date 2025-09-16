#!/usr/bin/env python3
from __future__ import annotations
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from src.model_embed import build_embed_only
from src.metrics_ext import summarize_scores, best_threshold_by_f1

def main():
    ap = argparse.ArgumentParser(description="Train embeddings-only phishing model (MiniLM + signals)")
    ap.add_argument("--csv", required=True, help="Training CSV with columns: subject,body,label[,cc]")
    ap.add_argument("--use_signals", action="store_true", help="Add handcrafted features (URLs, CC stats, etc.)")
    ap.add_argument("--alpha", type=float, default=0.1, help="(kept for compatibility; conformal not used in this minimal script)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Hold-out fraction")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ["subject", "body", "label"]:
        assert col in df.columns, f"Missing column: {col}"
    if "cc" not in df.columns:
        df["cc"] = ""

    y = df["label"].astype(int).to_numpy()
    X = df[["subject", "body", "cc"]]

    # time-aware split if you later add a 'timestamp' column; otherwise stratified:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    # split a calibration fold out of train for isotonic probability calibration
    Xfit, Xcal, yfit, ycal = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=123)

    base = build_embed_only(use_signals=args.use_signals)
    base.fit(Xfit, yfit)

    # Isotonic calibration over the already-fitted model
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(Xcal, ycal)

    # choose a decision threshold on calibration set by best F1
    cal_scores = cal.predict_proba(Xcal)[:, 1]
    thr = best_threshold_by_f1(ycal, cal_scores)

    # evaluate on untouched test split
    test_scores = cal.predict_proba(Xte)[:, 1]
    summary = summarize_scores(yte, test_scores, thr)
    print("\n==== TEST SUMMARY (Embeddings-only) ====")
    print(summary["report"])
    print({k: v for k, v in summary.items() if k != "report"})

    # save bundle
    out = Path("phish_embed_only.joblib")
    joblib.dump({"model": cal, "decision_threshold": float(thr), "version": 1}, out)
    print(f"Saved -> {out.resolve()}")

if __name__ == "__main__":
    main()
