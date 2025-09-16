# scripts/explain_global.py
import joblib, numpy as np

SIG_NAMES = ["n_urls","uniq_url_domains","n_exclam","caps_in_subject",
             "has_suspicious_phrase","is_internal_sender","internal_with_links",
             "n_links","n_imgs","has_form"]

bundle = joblib.load("phish_hybrid_calibrated.joblib")
model = bundle["model"]
ct = model.named_steps["ct"]; clf = model.named_steps["clf"]
tfidf = ct.transformers_[0][1].named_steps["tfidf"]
vocab = tfidf.get_feature_names_out()

coef = clf.coef_[0]
n_tfidf = len(vocab)
# adjust if your SBERT size differs
SBERT_DIM = 384
tfidf_w = coef[:n_tfidf]
signals_w = coef[n_tfidf + SBERT_DIM : n_tfidf + SBERT_DIM + len(SIG_NAMES)]

top = tfidf_w.argsort()[-30:][::-1]
print("\nTop + TF-IDF features:")
for i in top:
    print(f"{vocab[i]:30s}  {tfidf_w[i]:+.3f}")

print("\nSignal weights:")
for n, w in zip(SIG_NAMES, signals_w):
    print(f"{n:22s} {w:+.3f}")
