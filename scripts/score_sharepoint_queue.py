# # scripts/score_sharepoint_queue.py
# import os, time, joblib, pandas as pd
# from sp_token import get_sp_token
# from sp_client import SPClient

# SP_SITE   = os.environ["SP_SITE"]
# SP_LIST   = os.environ["SP_LIST"]            # e.g. PhishInbox
# MODEL     = os.environ.get("PHISH_MODEL", "phish_hybrid_calibrated.joblib")
# BATCH     = int(os.environ.get("SP_BATCH", "200"))
# INTERNAL_DOMAIN = os.environ.get("INTERNAL_DOMAIN", "")  # optional for features

# def now_utc():
#     return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# def as_frame(items):
#     rows = []
#     for it in items:
#         rows.append({
#             "subject": it.get("Subject") or "",
#             "body":    it.get("BodyPreview") or "",
#             "cc":      "",                 # map if you capture CC in Flow A
#             "from":    (it.get("From") or "").lower(),
#         })
#     return pd.DataFrame(rows)

# def main():
#     token = get_sp_token()
#     sp = SPClient(SP_SITE, token)

#     bundle = joblib.load(MODEL)
#     model  = bundle["model"]
#     ver    = bundle.get("model_version", "phish_hybrid_calibrated")

#     pending = sp.get_pending(SP_LIST, top=BATCH)
#     if not pending:
#         print("[info] no pending items"); return

#     X = as_frame(pending)
#     # Some pipelines only expect subject/body/cc; if you added 'from' feature it will be used.
#     probs = model.predict_proba(X)[:, 1]

#     when = now_utc()
#     for it, p in zip(pending, probs):
#         payload = {
#             "Score": float(p),
#             "Status": "Scored",
#             "ScoredAt": when,
#             "ModelVersion": ver,
#         }
#         try:
#             sp.update_item(SP_LIST, int(it["Id"]), payload)
#             print(f"[ok] item {it['Id']} score={p:.3f}")
#         except Exception as e:
#             print(f"[err] item {it['Id']} {e}")

# if __name__ == "__main__":
#     main()

# scripts/score_sharepoint_queue.py
# import os, time, joblib, pandas as pd, requests
# from sp_token import get_sp_token
# from sp_client import SPClient

# SP_SITE   = os.environ["SP_SITE"]
# SP_LIST   = os.environ["SP_LIST"]            # e.g. PhishInbox
# MODEL     = os.environ.get("PHISH_MODEL", "phish_hybrid_calibrated.joblib")
# MODEL_SP_PATH = os.environ.get("MODEL_SP_PATH", "")  # server-relative path on SP
# BATCH     = int(os.environ.get("SP_BATCH", "200"))

# def now_utc():
#     return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# def as_frame(items):
#     rows = []
#     for it in items:
#         rows.append({
#             "subject": it.get("Subject") or "",
#             "body":    it.get("BodyPreview") or "",
#             "cc":      "",                 # map if you capture CC in Flow A
#             "from":    (it.get("From") or "").lower(),
#         })
#     return pd.DataFrame(rows)

# def download_model_from_sharepoint(token, site_url, server_relative_path, out_path):
#     """
#     Downloads a file from SharePoint using the /_api/web/GetFileByServerRelativeUrl('<path>')/$value endpoint.
#     token - bearer token
#     site_url - https://contoso.sharepoint.com/sites/SiteName
#     server_relative_path - like /sites/SiteName/Shared Documents/Models/f.joblib
#     out_path - local path to save file
#     """
#     base = site_url.rstrip("/")
#     url = f"{base}/_api/web/GetFileByServerRelativeUrl('{server_relative_path}')/$value"
#     headers = {"Authorization": f"Bearer {token}"}
#     r = requests.get(url, headers=headers, stream=True, timeout=120)
#     r.raise_for_status()
#     with open(out_path, "wb") as fh:
#         for chunk in r.iter_content(chunk_size=1 << 20):
#             if chunk:
#                 fh.write(chunk)

# def ensure_model_present():
#     if os.path.exists(MODEL):
#         print(f"[info] model found locally: {MODEL}")
#         return

#     # if no MODEL_SP_PATH, warn and fail
#     if not MODEL_SP_PATH:
#         raise RuntimeError("Model missing locally and MODEL_SP_PATH not provided")

#     print("[info] model not found locally. Attempting to download from SharePoint...")
#     token = get_sp_token()
#     try:
#         download_model_from_sharepoint(token, SP_SITE, MODEL_SP_PATH, MODEL)
#         print(f"[ok] downloaded model -> {MODEL}")
#     except Exception as e:
#         raise RuntimeError(f"Failed to download model from SharePoint: {e}")

# def main():
#     # ensure model exists
#     ensure_model_present()

#     # create sp client and score
#     token = get_sp_token()
#     sp = SPClient(SP_SITE, token)

#     bundle = joblib.load(MODEL)
#     model  = bundle["model"]
#     ver    = bundle.get("model_version", "phish_hybrid_calibrated")

#     pending = sp.get_pending(SP_LIST, top=BATCH)
#     if not pending:
#         print("[info] no pending items"); return

#     X = as_frame(pending)
#     probs = model.predict_proba(X)[:, 1]

#     when = now_utc()
#     for it, p in zip(pending, probs):
#         payload = {
#             "Score": float(p),
#             "Status": "Scored",
#             "ScoredAt": when,
#             "ModelVersion": ver,
#         }
#         try:
#             sp.update_item(SP_LIST, int(it["Id"]), payload)
#             print(f"[ok] item {it['Id']} score={p:.3f}")
#         except Exception as e:
#             print(f"[err] item {it['Id']} {e}")

# if __name__ == "__main__":
#     main()


#Most updated:
import os
import time
import joblib
import pandas as pd
from datetime import datetime
from sp_client import SPClient

# Load environment variables
EMAIL = os.environ["SHAREPOINT_EMAIL"]
PASSWORD = os.environ["SHAREPOINT_PASSWORD"]
SITE = os.environ["SHAREPOINT_SITE"]
LIST_QUEUE = os.environ["SP_LIST1"]
LIST_REPORT = os.environ["SP_LIST2"]
MODEL_PATH = os.environ.get("PHISH_MODEL", "phish_hybrid_calibrated.joblib")
THRESHOLD = float(os.environ.get("PHISH_THRESHOLD", "0.7"))

# Initialize SharePoint client
sp = SPClient(SITE, EMAIL, PASSWORD)

# Load model
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
version = bundle.get("version", "unknown")

# Read pending items
pending_items = sp.get_pending(LIST_QUEUE, top=100)
if not pending_items:
    print("[info] No pending items found.")
    exit(0)

# Prepare data
rows = []
ids = []
for item in pending_items:
    rows.append({
        "subject": item.get("Title", ""),
        "body": item.get("Body", ""),
        "cc": item.get("CC", ""),
        "from": item.get("Sender", "")
    })
    ids.append((item["Id"], item.get("MessageID", "")))

X = pd.DataFrame(rows)
scores = model.predict_proba(X)[:, 1]
now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# Optional SHAP explanation stub
def get_explanation(model, X_row):
    return "Top features not implemented"

# Process each item
for (item_id, msg_id), score, row in zip(ids, scores, X.to_dict(orient="records")):
    prediction = "Phishing" if score >= THRESHOLD else "Safe"
    explanation = get_explanation(model, pd.DataFrame([row]))

    report_payload = {
        "MessageID": msg_id,
        "PhishingScore": round(float(score), 4),
        "Prediction": prediction,
        "SHAP_x0020_Explanation": explanation,
        "Status": "Scanned",
        "ReportedTime": now,
        "ModelVersion": version
    }

    update_payload = {
        "Status": "Scanned"
    }

    try:
        sp.create_item(LIST_REPORT, report_payload)
        sp.update_item(LIST_QUEUE, item_id, update_payload)
        print(f"[ok] Scored item {item_id} → {prediction} ({score:.3f})")
    except Exception as e:
        print(f"[err] Failed to process item {item_id}: {e}")

