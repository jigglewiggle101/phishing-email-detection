# Phishing Email Detection (ML + Power Automate)

## 📌 Overview

This repository provides a machine learning pipeline for detecting phishing emails using **hybrid embeddings + engineered signals**.
The project is designed to integrate seamlessly with **Microsoft Power Automate + SharePoint Online**, enabling real-time detection and quarantine of phishing messages in Outlook.

---

## ✨ Features

* **Hybrid Model**: Sentence-BERT embeddings + engineered features (URL counts, suspicious phrases, internal sender heuristics).
* **Calibrated Thresholding**: Tuned decision thresholds to balance **precision vs recall**.
* **Explainability**: Global feature importance & signal weights (via `explain_global.py`).
* **SharePoint Integration**: Emails scored through a SharePoint list (queue pattern).
* **GitHub Actions Workflow**: Automated batch scoring every 5 minutes.
* **Power Automate Bridge**: Flows for moving/quarantining emails in Outlook.

---

## 📂 Repository Structure

```
.github/workflows/
  phish_score.yml         # GitHub Actions workflow for SharePoint scoring
data/
  validation_phish_2024.csv   # small validation set
scripts/
  train_embed_only.py         # embeddings-only training
  train_hybrid_calibrated.py  # hybrid model training
  evaluate_on_validation.py   # eval against validation sets
  score_sharepoint_queue.py   # SharePoint scoring bridge
  set_threshold.py            # update model decision threshold
  explain_global.py           # explainability report
src/
  model_hybrid.py             # hybrid pipeline definition
  features_extra.py           # engineered feature extraction
sp_client.py / sp_token.py    # SharePoint REST/MSAL helpers
requirements.txt              # runtime dependencies
```

---

## 📊 Model Performance

* **Hybrid Model (40k train, 15k validation):**

  * F1 ≈ **0.97** (internal test)
  * ROC-AUC ≈ **0.995**
* **Validation (2024 dataset):**

  * ROC-AUC ≈ **0.946**
  * Best F1 ≈ **0.87** (at threshold ≈ 0.35)

---

## 🚀 Deployment Architecture

1. **Inbox → SharePoint (Power Automate Flow A)**

   * A new incoming Outlook email is written to a SharePoint list (`PhishInbox`) with `Status=Pending`.
2. **Scoring (GitHub Actions)**

   * Every 5 minutes, the workflow fetches pending rows, runs the ML model, and updates `Score` + `Status=Scored`.
3. **Quarantine/Report (Power Automate Flow B)**

   * Based on `Score` thresholds:

     * ≥ 0.70 → Move to Quarantine folder + Auto-report
     * 0.35 – 0.70 → Escalate to triage
     * < 0.35 → Deliver

---

## 🔧 Requirements

* Python 3.10+
* Dependencies:

  ```
  pip install -r requirements.txt
  ```
* SharePoint + Azure App Registration (service principal)
  Secrets required in GitHub:

  * `SP_TENANT_ID`, `SP_CLIENT_ID`, `SP_CLIENT_SECRET`, `SP_SITE`, `SP_LIST`, `MODEL_SP_PATH`

---

## 📥 Training & Evaluation

Train hybrid model:

```bash
export PYTHONPATH=$(pwd)
python scripts/train_hybrid_calibrated.py --csv data/combined_emails.csv --use_signals
```

Evaluate on validation:

```bash
python scripts/evaluate_on_validation.py \
  --model phish_hybrid_calibrated.joblib \
  --csv data/validation_phish_2024.csv --sweep
```

---

## ⚡ GitHub Actions Setup

1. Commit `.github/workflows/phish_score.yml`.
2. Upload model artifact (`phish_hybrid_calibrated.joblib`) to SharePoint (document library).
3. Add secrets in GitHub → Settings → Secrets → Actions:

   * `SP_TENANT_ID`, `SP_CLIENT_ID`, `SP_CLIENT_SECRET`, `SP_SITE`, `MODEL_SP_PATH`.
4. Trigger workflow manually or wait for cron (`*/5 * * * *`).

---

## 📌 Power Automate Flows

* **Flow A (Intake)**: Outlook → Create item in SharePoint list (`PhishInbox`).
* **Flow B (Action)**: On item modified (`Status=Scored`), move/quarantine/report based on `Score`.

---

## 🔮 Future Improvements

* Broader dataset coverage (multi-language, industry-specific phish).
* Richer signals (HTML form detection, attachment inspection).
* Explainability dashboards for triage analysts.
* Re-training automation with incremental data.

---

## 🛡️ Disclaimer

This project is provided for **educational and research purposes**.
Before deploying in production, review your organization’s compliance, governance, and security requirements.

---

