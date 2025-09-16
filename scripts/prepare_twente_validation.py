#!/usr/bin/env python3
import argparse, pandas as pd, re
from pathlib import Path

LABEL_MAP = {"phishing email": 1, "safe email": 0}

def split_subject_body(text: str, max_subj=120):
    if not isinstance(text, str): 
        return "", ""
    text = text.strip().replace("\r\n", "\n")
    # Heuristic 1: first newline = subject/body split
    if "\n" in text:
        subj, body = text.split("\n", 1)
        return subj.strip()[:max_subj], body.strip()
    # Heuristic 2: first sentence end
    m = re.search(r"([.!?])\s+", text)
    if m:
        cut = m.end()
        subj = text[:cut].strip()[:max_subj]
        body = text[cut:].strip()
        return subj, body
    # Fallback: first 120 chars as subject
    return text[:max_subj].strip(), text[max_subj:].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="CSV with 'Email Text','Email Type'")
    ap.add_argument("--out", dest="out_path", default="data/validation_phish_2024.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("email text") or cols.get("email_text") or list(df.columns)[0]
    type_col = cols.get("email type") or cols.get("email_type") or list(df.columns)[1]

    y = df[type_col].astype(str).str.strip().str.lower().map(LABEL_MAP)
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int)

    subs, bodies = [], []
    for t in df[text_col].fillna(""):
        s, b = split_subject_body(t)
        subs.append(s); bodies.append(b)

    out = pd.DataFrame({
        "subject": subs,
        "body": bodies,
        "label": y.values,
        "cc": ""    # no CC in this dataset
    })

    print(f"Rows: {len(out)} | phishing: {int(out['label'].sum())} | safe: {int((1-out['label']).sum())}")
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)
    print("Saved ->", args.out_path)

if __name__ == "__main__":
    main()
