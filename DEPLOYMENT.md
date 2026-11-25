# Streamlit Community Cloud deployment

The dashboard loads the bundled model and results using shared local Python inference. It does not require FastAPI, MLflow, or retraining at startup. `requirements.txt` is the pinned Cloud runtime manifest; Python 3.12 was used for verification.

Publication and deployment require explicit approval. GitHub CLI is authenticated, but no remote or target repository has been selected. No push or public deployment has been performed.

After approval: review the model, public data snapshots and source for publication; create or select the authorized GitHub repository; commit and push the reviewed branch; sign in at https://share.streamlit.io/, select the repository and branch, and set the main file path to `dashboard/app.py`. Allow Streamlit access to the repository. Verify the resulting hosted URL displays the real forecast, cutoff, comparison and recent historical data. A successful build alone is not verification.

The forecast is based on bundled data, so it becomes stale unless the model and snapshot are deliberately refreshed and redeployed. There is no automatic refresh or continuous training. Never publish credentials or MLflow local data. Unsloth is not required for classical regression and is excluded.
