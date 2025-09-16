# #!/usr/bin/env python3
# import argparse, joblib, pandas as pd
# from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", required=True)
#     ap.add_argument("--csv", required=True)
#     args = ap.parse_args()

#     bundle = joblib.load(args.model)
#     model = bundle["model"]
#     thr = float(bundle.get("decision_threshold", 0.5))

#     df = pd.read_csv(args.csv)
#     for col in ["subject","body","label"]:
#         assert col in df.columns
#     if "cc" not in df.columns:
#         df["cc"] = ""

#     X = df[["subject","body","cc"]]
#     y = df["label"].astype(int).values

#     p = model.predict_proba(X)[:,1]
#     yhat = (p >= thr).astype(int)

#     print("\n== Classification report ==")
#     print(classification_report(y, yhat, digits=3))
#     try:
#         print("PR-AUC:", round(average_precision_score(y, p), 4))
#         print("ROC-AUC:", round(roc_auc_score(y, p), 4))
#     except Exception:
#         pass

# if __name__ == "__main__":
#     main()
    

#!/usr/bin/env python3
import argparse, joblib, pandas as pd, numpy as np
from sklearn.metrics import (
    classification_report, average_precision_score, roc_auc_score,
    precision_recall_fscore_support
)

def sweep_thresholds(y_true, y_prob, start=0.2, end=0.8, steps=13):
    best = {"thr": None, "prec": 0, "rec": 0, "f1": -1}
    print("\n== Threshold sweep ==")
    for thr in np.linspace(start, end, steps):
        y_pred = (y_prob >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        print(f"th={thr:0.3f} | Prec={prec:0.3f} Rec={rec:0.3f} F1={f1:0.3f}")
        if f1 > best["f1"]:
            best = {"thr": float(thr), "prec": float(prec), "rec": float(rec), "f1": float(f1)}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="*.joblib")
    ap.add_argument("--csv", required=True, help="validation CSV with subject,body,label,cc")
    ap.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    ap.add_argument("--sweep", action="store_true", help="print metrics for a range of thresholds and best-F1")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    thr = float(args.threshold if args.threshold is not None else bundle.get("decision_threshold", 0.5))

    df = pd.read_csv(args.csv)
    for col in ["subject","body","label"]:
        assert col in df.columns, f"Missing column: {col}"
    if "cc" not in df.columns:
        df["cc"] = ""

    X = df[["subject","body","cc"]]
    y = df["label"].astype(int).values

    # scores & report at current threshold
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= thr).astype(int)

    print("\n== Classification report (thr={:.3f}) ==".format(thr))
    print(classification_report(y, yhat, digits=3))
    try:
        print("PR-AUC:", round(average_precision_score(y, p), 4))
        print("ROC-AUC:", round(roc_auc_score(y, p), 4))
    except Exception:
        pass

    if args.sweep:
        best = sweep_thresholds(y, p, start=0.2, end=0.8, steps=13)
        print("\nBest-F1 on this validation set:")
        print(f"thr={best['thr']:.3f} | Prec={best['prec']:.3f} Rec={best['rec']:.3f} F1={best['f1']:.3f}")

if __name__ == "__main__":
    main()
