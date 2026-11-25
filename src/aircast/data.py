"""Fetch/cache real hourly Chennai CAMS PM2.5 and weather data."""
from __future__ import annotations
import os
from pathlib import Path
import time
import hashlib
import pandas as pd
import requests

LATITUDE, LONGITUDE, TIMEZONE = 13.0827, 80.2707, "Asia/Kolkata"
URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
VARS = ["pm2_5", "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"]

def project_root() -> Path:
    return Path(os.environ.get("AIRCAST_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()

def fetch_hourly(start_date: str = "2022-01-01", end_date: str | None = None, *, force: bool = False) -> pd.DataFrame:
    """Fetch genuine hourly observations/forecasts; use a cache unless forced."""
    cache = project_root() / "data/raw/chennai_hourly.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, parse_dates=["time"])
        if start_date <= str(df.time.min().date()) and (end_date is None or str(df.time.max().date()) >= end_date):
            return df
    end_date = end_date or pd.Timestamp.now(tz=TIMEZONE).date().isoformat()
    session = requests.Session()
    params = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": ",".join(VARS), "timezone": TIMEZONE, "start_date": start_date, "end_date": end_date}
    payload = None
    for attempt in range(3):
        try:
            response = session.get(URL, params=params, timeout=45)
            response.raise_for_status(); payload = response.json(); break
        except requests.RequestException:
            if attempt == 2: raise
            time.sleep(2 ** attempt)
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly or "pm2_5" not in hourly:
        raise ValueError("Open-Meteo returned no hourly CAMS PM2.5 data")
    # Weather fields are absent from this endpoint (undefined units/all-null); retain only usable series.
    usable = {k: v for k, v in hourly.items() if k == "time" or (k in payload.get("hourly_units", {}) and payload["hourly_units"].get(k) != "undefined" and any(x is not None for x in v))}
    df = pd.DataFrame(usable)
    df["time"] = pd.to_datetime(df.time)
    df = df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    df.attrs["sha256"] = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    if df.pm2_5.isna().all(): raise ValueError("Open-Meteo returned no PM2.5 values")
    df.attrs["source"] = "Open-Meteo CAMS model-derived air-quality API"
    # Never replace a valid cache with an empty or malformed response.
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    df.to_csv(tmp, index=False); tmp.replace(cache)
    return df

def daily_from_hourly(hourly: pd.DataFrame, min_hours: int = 18) -> pd.DataFrame:
    """Aggregate complete-enough local calendar days; discard partial days."""
    frame = hourly.copy()
    frame["time"] = pd.to_datetime(frame.time)
    frame = frame.drop_duplicates("time", keep="last")
    frame["date"] = frame.time.dt.date
    numeric = [c for c in frame.columns if c not in {"time", "date"}]
    daily = frame.groupby("date")[numeric].agg("mean")
    daily["hours"] = frame.groupby("date").pm2_5.count()
    daily = daily.loc[daily.hours >= min_hours].dropna(subset=["pm2_5"])
    daily.index = pd.to_datetime(daily.index); daily.index.name = "date"
    return daily.reset_index().sort_values("date").reset_index(drop=True)

def load_daily(refresh: bool = False) -> pd.DataFrame:
    """Load daily dataset, refreshing from the actual Open-Meteo API as needed."""
    raw = fetch_hourly(force=refresh)
    daily = daily_from_hourly(raw)
    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).normalize()
    daily = daily[daily.date < now]
    if daily.empty: raise ValueError("No complete historical daily data available")
    if daily.date.max() < now - pd.Timedelta(days=3): raise ValueError("Latest complete input is stale; refresh Open-Meteo data")
    out = project_root() / "data/processed/daily.csv"
    out.parent.mkdir(parents=True, exist_ok=True); daily.to_csv(out, index=False)
    return daily
