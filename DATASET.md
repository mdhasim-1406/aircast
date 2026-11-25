# Dataset documentation

7,095 daily rows across Chennai, Delhi, Mumbai, Bengaluru, Kolkata, 2023-01-01 to2026-11-19; 1,419 each. Source endpoints: https://air-quality-api.open-meteo.com/v1/air-quality and https://archive-api.open-meteo.com/v1/archive . Timezone Asia/Kolkata. Pollutants µg/m³. ERA5 reanalysis weather values are historical estimates with availability lag; downloaded but not currently used by models.

CAMS is model-derived gridded atmospheric composition, not ground station data. Selected coordinates are preserved in providers.py; actual grid location can differ. Aggregation requires >=18 non-null unique hourly values. Real current snapshot retrieval uses latest timestamp no later than now. Cache TTL900seconds with retrieval time preserved. Targets in held-out periods are never imputed.

Actual snapshot validation found zero duplicated city-days, missing/negative PM2.5, incomplete days or flagged seven-day flatlines. Full missingness/coverage rules and timestamped provenance are artifacts/platform/quality.json and provenance.json. Dataset SHA256 is44cab407c5e17ced7efe45d141e50ef1c8931134b27b633f0b7d4dbe6e0c6d89. Do not remove real extreme pollution as generic outliers.

Attribution: Open-Meteo and Copernicus CAMS/ERA5. Review provider data licence and current non-commercial limits at https://open-meteo.com/en/terms and official air-quality documentation before commercial reuse. No OpenAQ data included because unauthenticated API access returned401. No synthetic training data; tests contain explicit deterministic fixtures only.
