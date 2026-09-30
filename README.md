# DEPIGrad: Fraud Detection with Explanations

A web app that scores a card transaction for fraud risk and tells you, in plain English, why it got that score. A gradient-boosted model does the scoring, SHAP finds the features that pushed the score the most, and Gemini turns the top 3 into a short explanation. If Gemini is unavailable, the app uses a template explanation instead, so the page never breaks.

This was my capstone for the DEPI Generative AI track.

**Live demo:** https://huggingface.co/spaces/3amtarekelgamd/depigrad

![Screenshot of the app](docs/screenshot.png)
<!-- TODO: add a screenshot at docs/screenshot.png -->

## How it works

1. **Data.** The model is trained on IEEE-CIS style transaction data (the Kaggle fraud dataset format: transaction, card, address, device and Vesta `V*` features). The final model takes 310 input features. Many of them are target-encoded (`*_te`) and there are some engineered ones like hour of day, amount z-scores and "was missing" flags.
2. **Preprocessing.** Inputs go through a saved quantile transformer (`fraud_quantile_transformer.pkl`). Any feature you leave out is filled with the median from a sample of the training data.
3. **Model.** The saved model is an XGBoost classifier (the metadata file names it "XGBoost (tuned, GPU)"). A decision threshold tuned during training decides fraud vs. not fraud. It is stored in `fraud_classifier_threshold_meta.pkl`.
4. **SHAP.** A SHAP `TreeExplainer` gives per-feature contributions for each prediction. The three features with the biggest absolute contribution are picked.
5. **Gemini explanation.** Those three feature names, the risk score and the fraud/not-fraud label go into a prompt, and Gemini writes a two-sentence explanation. Only the model output and feature names are sent, not raw transaction values.
6. **Fallback.** If there is no API key, or the Gemini call fails, the backend builds the explanation from a fixed template that names the same top features. The Gemini error is written to the server log, not shown in the UI.

## Results

These numbers are stored in `fraud_classifier_threshold_meta.pkl`, which was saved by the training run. I did not re-run the evaluation for this repo, and the training notebook is not included, so I can't show how the validation/test split was made.

| Set | ROC AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation | 0.9675 | 0.8876 | 0.7287 | 0.8004 |
| Test | 0.9709 | 0.8860 | 0.7448 | 0.8093 |

Threshold used: 0.5305 (same file).

## Tech stack

- Python, FastAPI, Uvicorn
- XGBoost (LightGBM is also in the requirements), scikit-learn, SHAP
- Google Gemini via the `google-genai` SDK
- Svelte 5 + Vite frontend
- Docker, deployed on Hugging Face Spaces

## Run it locally

You need Python 3.9+ and Node 18+.

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

For frontend development with hot reload, run `npm run dev` in `frontend/` and set the API endpoint field in the UI to `http://localhost:8000`.

## Run with Docker

```bash
docker build -t depigrad .
docker run -p 7860:7860 --env-file .env depigrad
```

Open http://localhost:7860. The image builds the frontend itself. If you don't have a `.env`, drop the `--env-file .env` part.

## API

- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` takes a JSON object of features (missing ones are filled in) and returns `probability`, `risk_score`, `is_fraud`, `top_risk_feature` and `ai_explanation`.

## Not finished / what I would improve

- **Training code is not in this repo.** The model files are here, but the notebook that trained them isn't, so the results above can't be reproduced from this repo alone. The training data (about 2.5 GB) is also left out.
- **The prompt tells Gemini to "sound like an advanced neural AI".** That's flashy wording, not accurate. The model is a gradient-boosted tree ensemble. The fallback text says "deep neural analysis" too, which is wrong for the same reason.
- **Only XGBoost is pinned.** `xgboost==3.0.5` is pinned because newer versions crash SHAP's `TreeExplainer` on this model. The other packages are unpinned, and scikit-learn prints a version warning when loading the transformer (saved with 1.6.1, tested with 1.7.2). I should pin everything.
- **Docker image not tested by me.** I ran the backend and the frontend build outside Docker, on CPU. The Dockerfile is what the Hugging Face Space uses.
- **`fraud_samples.csv` is unreliable.** It holds 5 rows labelled fraud, and in my test only 1 of them was flagged at the current threshold. `demo_fraud_data.csv` (10 rows, all flagged as fraud) is the better demo file.
- **No proper tests.** `test_backend.py` and `test_gemini.py` are manual scripts, not a test suite.
- **Recall is 0.74 on the test set.** About a quarter of fraud cases are missed at the current threshold. The threshold could be moved depending on how costly a miss is compared to a false alarm.
- **CORS is open to all origins** and there is no rate limiting. Fine for a demo, not for production.
- **The repo has some leftover scratch scripts** (`patch_svelte_*.py`, `extract_frauds.py` and similar) from building the demo samples.

## Project layout

```
app.py                  FastAPI backend (prediction, SHAP, Gemini, fallback)
*.pkl                   trained model, quantile transformer, threshold + metrics
feature_config.json     feature names and medians used to fill missing inputs
frontend/               Svelte app
Dockerfile              builds frontend, runs the API on port 7860
```
