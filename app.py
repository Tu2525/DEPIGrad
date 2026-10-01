import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any
import json
import os
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    # 10 s timeout so a slow Gemini call can't hold a worker; the fallback text covers it
    gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=10_000))
else:
    gemini_client = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("fraud_classifier_phase3.pkl")
transformer = joblib.load("fraud_quantile_transformer.pkl")
explainer = shap.TreeExplainer(model)
threshold_meta = joblib.load("fraud_classifier_threshold_meta.pkl")
fraud_threshold = float(threshold_meta.get("threshold", 0.80))

with open("feature_config.json", "r") as f:
    config = json.load(f)
    FEATURE_NAMES = config["feature_names"]
    FEATURE_MEDIANS = config["medians"]

# Convert QT Medians to RAW Medians so we don't double scale values.
# No fallback on purpose: using the QT medians as raw values would silently skew every prediction.
medians_array = np.array([[FEATURE_MEDIANS[f] for f in FEATURE_NAMES]], dtype=float)
raw_medians_array = transformer.inverse_transform(medians_array)
RAW_MEDIANS = {f: float(raw_medians_array[0][i]) for i, f in enumerate(FEATURE_NAMES)}

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Dict[str, Any]):
    # Assemble the 310 feature array
    input_features = []
    for feature in FEATURE_NAMES:
        val = transaction.get(feature)
        if val is None or val == "":
            val = RAW_MEDIANS.get(feature, 0.0)
        try:
            input_features.append(float(val))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"Feature '{feature}' must be a number, got {val!r}")
    
    input_array = np.array([input_features], dtype=float)
    scaled_array = transformer.transform(input_array)
    probability = float(model.predict_proba(scaled_array)[0][1])
    risk_score = int(round(probability * 100))
    is_fraud = probability > fraud_threshold

    shap_values = explainer.shap_values(scaled_array)
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[-1])[0]
    else:
        shap_row = np.asarray(shap_values)[0]

    top_indices = np.argsort(np.abs(shap_row))[-3:][::-1]
    top_features = [FEATURE_NAMES[i] for i in top_indices]
    top_risk_feature = top_features[0]

    if is_fraud:
        fallback_explanation = (
            f"The model flagged this transaction mainly because of the values of '{top_features[0]}', "
            f"followed by '{top_features[1]}' and '{top_features[2]}'. "
            "This combination is strongly associated with fraud in the training data."
        )
    else:
        fallback_explanation = (
            "The model scored this transaction as low risk. "
            f"'{top_features[0]}' and '{top_features[1]}' influenced the score the most, "
            "but not enough to indicate fraud."
        )

    explanation = fallback_explanation

    if gemini_client:
        prompt = (
            "You are a fraud analyst. A gradient-boosted tree model (XGBoost) scored a card transaction:\n"
            f"- Risk score: {risk_score}%\n"
            f"- Decision: {'Fraud' if is_fraud else 'Not fraud'}\n"
            f"- Top contributing features by SHAP value: {top_features[0]}, {top_features[1]}, and {top_features[2]}.\n\n"
            "Write a concise, professional 2-sentence explanation of this decision for the app's user interface. "
            "Use only the information above and name the top features. Do not invent values, account history "
            "or spending patterns, and do not claim more certainty than the score shows. Plain text only, no markdown."
        )
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            if response and response.text:
                explanation = response.text.strip()
        except Exception as e:
            logging.warning("Gemini call failed, using fallback explanation: %s", e)

    return {
        "probability": probability,
        "risk_score": risk_score,
        "is_fraud": is_fraud,
        "top_risk_feature": top_risk_feature,
        "ai_explanation": explanation,
    }

# Serve the built frontend if present (tests and API-only dev run without it)
if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

