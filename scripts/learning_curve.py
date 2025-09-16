# #!/usr/bin/env python3
# from __future__ import annotations
# import argparse, os, time, joblib, numpy as np, pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# import matplotlib.pyplot as plt

# # local imports
# import sys, pathlib
# sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# from src.model_hybrid import build_hybrid

# def metric_bundle(y_true, y_prob, thr=0.5):
#     y_pred = (y_prob >= thr).astype(int)
#     return {
#         "f1": f1_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred),
#         "recall": recall_score(y_true, y_pred),
#         "accuracy": accuracy_score(y_true, y_pred),
#     }

# def main():
#     ap = argparse.ArgumentParser(description="Learning curve for Hybrid (TF-IDF + SBERT + Signals)")
#     ap.add_argument("--csv", required=True, help="Training CSV with subject,body,label[,cc]")
#     ap.add_argument("--out_png", default="learning_curve.png")
#     ap.add_argument("--use_signals", action="store_true", help="include handcrafted signals")
#     ap.add_argument("--max_features", type=int, default=150_000)
#     ap.add_argument("--min_df", type=int, default=2)
#     ap.add_argument("--test_size", type=float, default=0.2)
#     ap.add_argument("--sizes", default="1000,5000,10000,20000,40000", help="train sizes (absolute counts) to evaluate")
#     ap.add_argument("--thr", type=float, default=0.5, help="decision threshold used for metrics")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--max_seq_len", type=int, default=128, help="SBERT max tokens (speed vs perf)")
#     ap.add_argument("--batch", type=int, default=int(os.getenv("SBERT_BATCH", "256")))
#     ap.add_argument("--limit", type=int, default=None, help="only first N rows for quick tests")
#     args = ap.parse_args()

#     # environment (speed)
#     os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
#     os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
#     os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
#     os.environ.setdefault("SBERT_BATCH", str(args.batch))

#     df = pd.read_csv(args.csv, low_memory=False)
#     if args.limit: df = df.head(args.limit)
#     for col in ["subject","body","label"]:
#         assert col in df.columns, f"Missing column: {col}"
#     if "cc" not in df.columns:
#         df["cc"] = ""

#     X_all = df[["subject","body","cc"]]
#     y_all = df["label"].astype(int).to_numpy()

#     # fixed validation split (hold-out)
#     Xtr_full, Xval, ytr_full, yval = train_test_split(
#         X_all, y_all, test_size=args.test_size, stratify=y_all, random_state=args.seed
#     )

#     # sizes to evaluate
#     sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
#     sizes = [s for s in sizes if s <= len(Xtr_full)]
#     if not sizes:
#         raise SystemExit("No valid sizes to evaluate (check --sizes vs dataset size).")

#     results = []

#     # Build a base pipeline with your settings
#     # NOTE: we pass smaller max_features and max_seq_len for speed; tweak as needed
#     def make_model():
#         # Patch SBertEncoder defaults via environment: SBERT_BATCH; we also set max_seq_len via monkeypatch
#         # We’ll set on the instance right after creation.
#         model = build_hybrid(
#             max_features=args.max_features,
#             min_df=args.min_df,
#             use_signals=args.use_signals,
#         )
#         # set SBERT max_seq_len inside the pipeline (find the step)
#         try:
#             sbert = model.named_steps["pre"].transformers_[1][1].named_steps["sbert"]  # ("sbert", Pipeline(...))
#             sbert.max_seq_len = args.max_seq_len  # our SBertEncoder accepts this attr
#         except Exception:
#             pass
#         return model

#     print(f"[info] Hold-out val size: {len(Xval)} | Train pool: {len(Xtr_full)}")
#     print(f"[info] Train sizes: {sizes}")
#     print(f"[info] Using threshold={args.thr}")

#     for n in sizes:
#         # sample n from the training pool (stratified)
#         Xsub, _, ysub, _ = train_test_split(
#             Xtr_full, ytr_full, train_size=n, stratify=ytr_full, random_state=args.seed
#         )

#         model = make_model()
#         t0 = time.perf_counter()
#         model.fit(Xsub, ysub)
#         t_fit = time.perf_counter() - t0

#         # training metrics
#         p_tr = model.predict_proba(Xsub)[:,1]
#         mb_tr = metric_bundle(ysub, p_tr, thr=args.thr)

#         # validation metrics (fixed hold-out)
#         p_val = model.predict_proba(Xval)[:,1]
#         mb_val = metric_bundle(yval, p_val, thr=args.thr)

#         results.append({
#             "n": n, "fit_sec": t_fit,
#             "train_f1": mb_tr["f1"], "train_prec": mb_tr["precision"], "train_rec": mb_tr["recall"], "train_acc": mb_tr["accuracy"],
#             "val_f1": mb_val["f1"], "val_prec": mb_val["precision"], "val_rec": mb_val["recall"], "val_acc": mb_val["accuracy"],
#         })
#         print(f"[{n:>6}] fit {t_fit:6.1f}s | train F1 {mb_tr['f1']:.3f} | val F1 {mb_val['f1']:.3f}")

#     res = pd.DataFrame(results)
#     csv_out = Path(args.out_png).with_suffix(".csv")
#     res.to_csv(csv_out, index=False)
#     print("Saved metrics ->", csv_out.resolve())

#     # plot
#     plt.figure(figsize=(8,5))
#     plt.plot(res["n"], res["train_f1"], marker="o", label="Train F1")
#     plt.plot(res["n"], res["val_f1"], marker="o", label="Validation F1")
#     plt.xlabel("Training size (emails)")
#     plt.ylabel("F1 score")
#     plt.title("Learning Curve — Hybrid (TF-IDF + SBERT + Signals)")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(args.out_png, dpi=160)
#     print("Saved plot ->", Path(args.out_png).resolve())

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, time, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# local imports
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.model_hybrid import build_hybrid

def metric_bundle(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }

def main():
    ap = argparse.ArgumentParser(description="Learning curve for Hybrid (TF-IDF + SBERT + Signals)")
    ap.add_argument("--csv", required=True, help="Training CSV with subject,body,label[,cc]")
    ap.add_argument("--out_png", default="learning_curve.png")
    ap.add_argument("--use_signals", action="store_true", help="include handcrafted signals")
    ap.add_argument("--max_features", type=int, default=150_000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--sizes", default="1000,5000,10000,20000,40000", help="train sizes (absolute counts) to evaluate")
    ap.add_argument("--thr", type=float, default=0.5, help="decision threshold used for metrics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seq_len", type=int, default=128, help="SBERT max tokens (speed vs perf)")
    ap.add_argument("--batch", type=int, default=int(os.getenv("SBERT_BATCH", "256")))
    ap.add_argument("--limit", type=int, default=None, help="only first N rows for quick tests")
    args = ap.parse_args()

    # environment (speed)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("SBERT_BATCH", str(args.batch))

    df = pd.read_csv(args.csv, low_memory=False)
    if args.limit: df = df.head(args.limit)
    for col in ["subject","body","label"]:
        assert col in df.columns, f"Missing column: {col}"
    if "cc" not in df.columns:
        df["cc"] = ""

    X_all = df[["subject","body","cc"]]
    y_all = df["label"].astype(int).to_numpy()

    # fixed validation split (hold-out)
    Xtr_full, Xval, ytr_full, yval = train_test_split(
        X_all, y_all, test_size=args.test_size, stratify=y_all, random_state=args.seed
    )

    # sizes to evaluate
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    sizes = [s for s in sizes if s <= len(Xtr_full)]
    if not sizes:
        raise SystemExit("No valid sizes to evaluate (check --sizes vs dataset size).")

    results = []

    # Build a base pipeline with your settings
    # NOTE: we pass smaller max_features and max_seq_len for speed; tweak as needed
    def make_model():
        # Patch SBertEncoder defaults via environment: SBERT_BATCH; we also set max_seq_len via monkeypatch
        # We’ll set on the instance right after creation.
        model = build_hybrid(
            max_features=args.max_features,
            min_df=args.min_df,
            use_signals=args.use_signals,
        )
        # set SBERT max_seq_len inside the pipeline (find the step)
        try:
            sbert = model.named_steps["pre"].transformers_[1][1].named_steps["sbert"]  # ("sbert", Pipeline(...))
            sbert.max_seq_len = args.max_seq_len  # our SBertEncoder accepts this attr
        except Exception:
            pass
        return model

    print(f"[info] Hold-out val size: {len(Xval)} | Train pool: {len(Xtr_full)}")
    print(f"[info] Train sizes: {sizes}")
    print(f"[info] Using threshold={args.thr}")

    for n in sizes:
        # sample n from the training pool (stratified)
        Xsub, _, ysub, _ = train_test_split(
            Xtr_full, ytr_full, train_size=n, stratify=ytr_full, random_state=args.seed
        )

        model = make_model()
        t0 = time.perf_counter()
        model.fit(Xsub, ysub)
        t_fit = time.perf_counter() - t0

        # training metrics
        p_tr = model.predict_proba(Xsub)[:,1]
        mb_tr = metric_bundle(ysub, p_tr, thr=args.thr)

        # validation metrics (fixed hold-out)
        p_val = model.predict_proba(Xval)[:,1]
        mb_val = metric_bundle(yval, p_val, thr=args.thr)

        results.append({
            "n": n, "fit_sec": t_fit,
            "train_f1": mb_tr["f1"], "train_prec": mb_tr["precision"], "train_rec": mb_tr["recall"], "train_acc": mb_tr["accuracy"],
            "val_f1": mb_val["f1"], "val_prec": mb_val["precision"], "val_rec": mb_val["recall"], "val_acc": mb_val["accuracy"],
        })
        print(f"[{n:>6}] fit {t_fit:6.1f}s | train F1 {mb_tr['f1']:.3f} | val F1 {mb_val['f1']:.3f}")

    res = pd.DataFrame(results)
    csv_out = Path(args.out_png).with_suffix(".csv")
    res.to_csv(csv_out, index=False)
    print("Saved metrics ->", csv_out.resolve())

    # plot
    plt.figure(figsize=(8,5))
    plt.plot(res["n"], res["train_f1"], marker="o", label="Train F1")
    plt.plot(res["n"], res["val_f1"], marker="o", label="Validation F1")
    plt.xlabel("Training size (emails)")
    plt.ylabel("F1 score")
    plt.title("Learning Curve — Hybrid (TF-IDF + SBERT + Signals)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    print("Saved plot ->", Path(args.out_png).resolve())

if __name__ == "__main__":
    main()
