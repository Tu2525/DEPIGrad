from huggingface_hub import HfApi

print("Starting FINAL STRICT deployment to Hugging Face Spaces...")
api = HfApi()

# Absolutely no glob patterns except in frontend/src
allowed = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "fraud_classifier_phase3.pkl",
    "fraud_quantile_transformer.pkl",
    "fraud_classifier_threshold_meta.pkl",
    "feature_config.json",
    "feature_medians.json",
    "frontend/src/*",
    "frontend/src/**/*",
    "frontend/static/*",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/vite.config.js",
    "frontend/svelte.config.js",
    "frontend/index.html",
    "frontend/CLOUDFLARE_PAGES.md",
    "frontend/dist/**/*"
]

api.upload_folder(
    folder_path=".",
    repo_id="3amtarekelgamd/depigrad",
    repo_type="space",
    allow_patterns=allowed,
    commit_message="Deploy Gemini Gen AI Feature (Perfect Scope)"
)
print("Deployment completed successfully!")
