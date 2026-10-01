# DEPIGrad: Fraud Detection with Explanations

A web app that scores a card transaction for fraud risk and tells you, in plain English, why it got that score. A gradient-boosted model does the scoring, SHAP finds the features that pushed the score the most, and Gemini turns the top 3 into a short explanation. If Gemini is unavailable, the app uses a template explanation instead, so the page never breaks.

This was my capstone for the DEPI Generative AI track.

**Live demo:** https://huggingface.co/spaces/3amtarekelgamd/depigrad (it sleeps when idle, so the first load can take a minute)

## How it works

1. **Data.** The model is trained on IEEE-CIS style transaction data (the Kaggle fraud dataset format: transaction, card, address, device and Vesta `V*` features). The final model takes 310 input features. Many of them are target-encoded (`*_te`) and there are some engineered ones like hour of day, amount z-scores and "was missing" flags.
2. **Preprocessing.** Inputs go through a saved quantile transformer (`fraud_quantile_transformer.pkl`). Any feature you leave out is filled with the median from a sample of the training data.
3. **Model.** The saved model is an XGBoost classifier (the metadata file names it "XGBoost (tuned, GPU)"). A decision threshold tuned during training decides fraud vs. not fraud. It is stored in `fraud_classifier_threshold_meta.pkl`.
4. **SHAP.** A SHAP `TreeExplainer` gives per-feature contributions for each prediction. The three features with the biggest absolute contribution are picked.
5. **Gemini explanation.** Those three feature names, the risk score and the fraud/not-fraud label go into a prompt, and Gemini writes a two-sentence explanation. Only the model output and feature names are sent, not raw transaction values.
6. **Fallback.** If there is no API key, or the Gemini call fails or takes longer than 10 seconds, the backend builds the explanation from a fixed template that names the same top features. The Gemini error is written to the server log, not shown in the UI.

## Results

These numbers are stored in `fraud_classifier_threshold_meta.pkl`, which was saved by the training run. I did not re-run the evaluation for this repo, and the training notebook is not included, so I can't show how the validation/test split was made.

| Set | ROC AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation | 0.9675 | 0.8876 | 0.7287 | 0.8004 |
| Test | 0.9709 | 0.8860 | 0.7448 | 0.8093 |

Threshold used: 0.5305 (same file).

## Tech stack

- Python, FastAPI, Uvicorn
- XGBoost, scikit-learn, SHAP
- Google Gemini via the `google-genai` SDK
- Svelte 5 + Vite frontend
- pytest, GitHub Actions
- Docker, deployed on Hugging Face Spaces

## Run it locally

You need Python 3.11+ (tested on 3.13) and Node 20+.

```bash
git clone https://github.com/Tu2525/depigrad.git
cd depigrad

# Backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend (the backend serves the built files from frontend/dist)
cd frontend
npm install
npm run build
cd ..

# Optional: enable Gemini explanations
cp .env.example .env             # then put your key in GEMINI_API_KEY
```

Then start the server:

```bash
uvicorn app:app --port 8000
```

Open http://localhost:8000. Without a Gemini key, everything works but you get the template explanation.

For frontend development with hot reload, run `npm run dev` in `frontend/` and set the API endpoint field in the UI to `http://localhost:8000`. The backend also runs without a frontend build, serving just the API.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the health check, filling in missing features, rejecting non-numeric input with a 422, and checking that all 10 known fraud rows in `demo_fraud_data.csv` are flagged. Gemini is switched off during tests. GitHub Actions runs them on every push.

## Run with Docker

```bash
docker build -t depigrad .
docker run -p 7860:7860 --env-file .env depigrad
```

Open http://localhost:7860. The image builds the frontend in a separate Node stage, so the final image has no Node in it. If you don't have a `.env`, drop the `--env-file .env` part.

To deploy to the Hugging Face Space, run `python scripts/deploy_hf.py` and set `GEMINI_API_KEY` as a secret in the Space settings.

## API

- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` takes a JSON object of features (missing ones are filled in) and returns `probability`, `risk_score`, `is_fraud`, `top_risk_feature` and `ai_explanation`. A feature value that isn't a number returns a 422 that names the feature.

## Not finished / what I would improve

- **Training code is not in this repo.** The model files are here, but the notebook that trained them isn't, so the results above can't be reproduced from this repo alone. The training data (about 2.5 GB) is also left out.
- **The model is stored as a pickle.** XGBoost prints a warning when loading it because it was pickled with an older XGBoost version. The fix is to re-export it with `Booster.save_model` (JSON) from the training environment. scikit-learn is pinned to 1.6.1, the version the transformer was saved with.
- **Recall is 0.74 on the test set.** About a quarter of fraud cases are missed at the current threshold. The threshold could be moved depending on how costly a miss is compared to a false alarm.
- **CORS is open to all origins** and there is no rate limiting or auth. Fine for a demo, not for production.

## Project layout

```
app.py                  FastAPI backend (prediction, SHAP, Gemini, fallback)
test_app.py             API tests (pytest)
*.pkl                   trained model, quantile transformer, threshold + metrics
feature_config.json     feature names and medians used to fill missing inputs
demo_fraud_data.csv     10 known fraud rows, used by the tests and for trying the CSV upload
frontend/               Svelte app
scripts/                regenerate feature_config.json / demo data, deploy to Hugging Face
Dockerfile              Node stage builds the frontend, Python stage runs the API on port 7860
```
