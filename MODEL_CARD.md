# Release model card

Task: end-of-IST-day next-day mean PM2.5, µg/m³. Version 20261125T160022Z. Coverage: Bengaluru, Chennai, Delhi, Kolkata, Mumbai only. Target: Open-Meteo CAMS model-derived PM2.5, not independent ground truth.

Per-city validation-selected Ridge or persistence point estimator; separate LightGBM .1/.5/.9 quantile estimators with conformal expansion. Strictly chronological calibration and test. Intervals describe model-derived targets and are not guaranteed confidence intervals. Nominal80% actual coverage varies68.2%-95.3% across cities; see README and saved CSVs.

Not for health decisions. No future actual pollution/weather features. No station validation. Weather/city/Diwali covariates not integrated. Temporal shift and high-pollution spikes remain important risks. No demonstrated tenfold scaling or production reliability. Current release uses common temporal cutoffs and separate per-city models, not a nationwide transferable model. Validation-refit diagnostics are not unbiased holdout evidence. Model may be stale if recent provider history cannot be acquired; inference must refuse or visibly explain this.
