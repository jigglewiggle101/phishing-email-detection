#!/usr/bin/env python3
from __future__ import annotations
import argparse, joblib, numpy as np, pandas as pd, time, os, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from src.model_hybrid import build_hybrid
from src.metrics_ext import summarize_scores, best_threshold_by_f1

def main():
    ap = argparse.ArgumentParser(description="Train Hybrid (TF-IDF + SBERT + Signals) with isotonic calibration")
    ap.add_argument("--csv", required=True, help="Training CSV with subject,body,label[,cc]")
    ap.add_argument("--max_features", type=int, default=200_000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--use_signals", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    for c in ["subject","body","label"]:
        assert c in df.columns, f"Missing column: {c}"
    if "cc" not in df.columns:
        df["cc"] = ""
    X = df[["subject","body","cc"]]
    y = df["label"].astype(int).to_numpy()

    # (optional) speed on CPU
    try:
        torch.set_num_threads(os.cpu_count() or 8)
    except Exception:
        pass

    # splits
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    Xfit, Xcal, yfit, ycal = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=123)

    print(f"[info] fit={len(Xfit)}  cal={len(Xcal)}  test={len(Xte)}")
    t0 = time.perf_counter()
    base = build_hybrid(max_features=args.max_features, min_df=args.min_df, use_signals=args.use_signals)
    base.fit(Xfit, yfit)
    print(f"[done] base fit in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(Xcal, ycal)
    print(f"[done] calibration in {time.perf_counter()-t1:.1f}s")

    cal_scores = cal.predict_proba(Xcal)[:,1]
    thr = best_threshold_by_f1(ycal, cal_scores)

    test_scores = cal.predict_proba(Xte)[:,1]
    summary = summarize_scores(yte, test_scores, thr)
    print("\n==== TEST SUMMARY (Hybrid) ====")
    print(summary["report"])
    print({k:v for k,v in summary.items() if k!="report"})

    out = Path("phish_hybrid_calibrated.joblib")
    joblib.dump({"model": cal, "decision_threshold": float(thr), "version": 1}, out)
    print(f"Saved -> {out.resolve()}")

if __name__ == "__main__":
    main()
