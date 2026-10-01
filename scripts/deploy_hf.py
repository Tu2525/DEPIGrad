# Upload the app to the Hugging Face Space, which rebuilds it from the Dockerfile.
# Run from the repo root: python scripts/deploy_hf.py (needs `huggingface-cli login` first).
from huggingface_hub import HfApi

HfApi().upload_folder(
    folder_path=".",
    repo_id="3amtarekelgamd/depigrad",
    repo_type="space",
    allow_patterns=[
        "app.py",
        "requirements.txt",
        "Dockerfile",
        "fraud_classifier_phase3.pkl",
        "fraud_classifier_threshold_meta.pkl",
        "fraud_quantile_transformer.pkl",
        "feature_config.json",
        "demo_fraud_data.csv",
        "frontend/src/**",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/vite.config.js",
        "frontend/svelte.config.js",
        "frontend/index.html",
    ],
    # Files from earlier deploys that the app no longer uses
    delete_patterns=["App.svelte", "DEPLOYMENT.md", "fraud_samples.csv", "feature_medians.json"],
    commit_message="Deploy from GitHub repo",
)
print("Uploaded. The Space will rebuild.")
