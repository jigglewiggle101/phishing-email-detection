from __future__ import annotations
import os, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("PHISH_MODEL", "phish_hybrid_calibrated.joblib")
app = FastAPI(title="phish-hybrid")

BUNDLE = joblib.load(MODEL_PATH)
MODEL = BUNDLE["model"]
THR = float(BUNDLE.get("decision_threshold", 0.5))

class EmailIn(BaseModel):
    subject: str
    body: str = ""
    cc: str = ""

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(email: EmailIn):
    X = pd.DataFrame([{"subject": email.subject, "body": email.body, "cc": email.cc}])
    p = float(MODEL.predict_proba(X)[:,1][0])
    return {"score": round(p,4), "label": int(p>=THR), "threshold": THR}
