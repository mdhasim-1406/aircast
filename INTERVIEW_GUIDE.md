# AirCast interview guide

## Forecasting contract
At the end of a completed IST day t, predict the next IST day's average PM2.5 using information available through t. A provider forecast is not the custom ML forecast. CAMS targets represent atmospheric model estimates, not necessarily local ground measurements.

## Why persistence matters
Tomorrow equals today's value is cheap, causal and often strong because pollution is autocorrelated. A model must justify its complexity against this baseline and the seven-day seasonal baseline on held-out chronological data. Winning a validation comparison does not guarantee future superiority.

## Temporal leakage
Random splits allow future seasonal regimes and overlapping windows into training. Chronological cutoffs better approximate deployment. Fit preprocessing on training only; use past-only lags/rolling features and no future realized weather. Synchronize cutoffs across cities so another city's future cannot leak into pooled training. Never impute held-out target outcomes. Check target dates, not merely row ordering.

## Quantile regression and pinball loss
P10/P50/P90 estimate conditional quantiles, not a Gaussian confidence interval. Pinball loss penalizes quantile-specific over/underprediction asymmetrically. Quantile crossing must be handled so lower ≤ median ≤ upper. An interval must be assessed for both empirical coverage and width; a wide interval can achieve coverage without being useful.

## Conformal calibration
A separate later calibration period estimates how much predicted intervals miss the targets and adjusts their bounds. Do not tune on the final test. Nominal 80% coverage is not a guarantee under time-series dependence or distribution shift, and per-city coverage can differ from pooled coverage. Report actual held-out coverage.

## Overfitting and underfitting
Compare training with validation errors, rolling-origin stability and baseline errors. Low training error with worse validation suggests overfitting or temporal shift; high error on both can indicate insufficient features or model capacity. Diagnose seasonal and high-pollution cases rather than blindly removing extreme values. Use regularization and bounded tuning before increasing complexity.

## Drift versus model degradation
Feature drift concerns changes in input distributions; performance degradation requires subsequently observed outcomes. KS/PSI thresholds need minimum sample sizes. A drift alert alone does not prove worse forecasting, while unchanged inputs do not prove good performance. Backtests are not live monitoring; an empty forecast/outcome ledger must say insufficient data.

## Shared inference and deployment
Both FastAPI and Streamlit should load the same validated bundle and use identical feature generation. Streamlit loads pretrained artifacts without calling a laptop or retraining on startup. Versioned directories and an atomic manifest preserve the previous valid release if candidate generation fails. MLflow records experiments, not automatic scientific validity.

## Ten times more data
First retain compact Parquet, batched provider access and bounded CPU training. Profile ingestion/training/memory before introducing services. Increase historical/city validation coverage, monitor missingness and maintain reproducible dataset versions. Do not claim horizontal scaling without measuring concurrent usage and state persistence.

See README and model artifacts for actual release metrics. This explanation is methodology, not evidence that every capability has passed acceptance.
