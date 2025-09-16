#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, hashlib, pandas as pd, numpy as np
from pathlib import Path

CSV_CANDIDATES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "PhishingEmailData.csv",
    "SpamAssassin.csv",
    # NOTE: intentionally skipping spamResults.csv (no labels)
]

def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None

def normalize_one(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    csubj = pick_col(df, ["subject","email_subject"])
    cbody = pick_col(df, ["body","email_content","text","message"])
    clabel = pick_col(df, ["label","is_spam","is_phishing","class","category","result"])
    ccc = pick_col(df, ["cc"])  # rarely present

    subj = df[csubj] if csubj else ""
    body = df[cbody] if cbody else ""
    # map labels to {0,1}
    if clabel is not None:
        y = df[clabel]
        y = y.map(lambda v: 1 if str(v).strip().lower() in {"1","phishing","spam","true","phish","fraud","yes"} 
                         else (0 if str(v).strip().lower() in {"0","ham","false","safe","no"} else np.nan))
        if y.isna().mean() > 0.5:
            try:
                y = df[clabel].astype(int)
            except Exception:
                pass
    else:
        y = pd.Series(np.nan, index=df.index)

    cc = (df[ccc] if ccc else pd.Series([""]*len(df))).fillna("").astype(str)

    out = pd.DataFrame({
        "subject": subj.fillna("").astype(str).str.strip(),
        "body": body.fillna("").astype(str),
        "label": y,
        "cc": cc,
        "_source": source_name
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="data/combined_emails.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    frames = []
    for name in CSV_CANDIDATES:
        p = data_dir / name
        if not p.exists(): 
            print(f"[skip] {name} not found")
            continue
        df = read_csv_any(p)
        frames.append(normalize_one(df, name))
        print(f"[ok] {name}: {len(df)} rows")

    if not frames:
        raise SystemExit("No input CSVs found.")

    dfc = pd.concat(frames, ignore_index=True)
    # drop rows with missing label or empty subject+body
    dfc = dfc.dropna(subset=["label"])
    dfc["label"] = dfc["label"].astype(int).clip(0,1)
    dfc = dfc[~((dfc["subject"]=="") & (dfc["body"]==""))].copy()

    # dedupe near-identical by normalized text hash
    def norm_text(s): return re.sub(r"\s+", " ", s.lower()).strip()
    dfc["_norm"] = (dfc["subject"] + "\n" + dfc["body"]).map(norm_text)
    dfc["_hash"] = dfc["_norm"].apply(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    before = len(dfc)
    dfc = dfc.drop_duplicates(subset=["_hash"]).copy()
    after = len(dfc)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfc[["subject","body","label","cc","_source"]].to_csv(out_path, index=False)

    print("\n=== Build summary ===")
    print(f"rows before dedup: {before}")
    print(f"rows after  dedup: {after}")
    print("label counts:", dfc["label"].value_counts().to_dict())
    print("saved ->", out_path.resolve())

if __name__ == "__main__":
    main()
