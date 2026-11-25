"""Typed API over the shared provider, probabilistic inference, and ledger."""
from __future__ import annotations
from functools import lru_cache
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from aircast.contracts import AirQualitySnapshot, CityLocation, ForecastResult, MonitoringReport
from aircast.inference import InferenceError, load_inference
from aircast.monitoring import build_report, record_forecast
from aircast.providers import curated_cities

app = FastAPI(title="AirCast Platform API", version="2.0.0")

class PredictRequest(BaseModel):
    city_id: str = Field(min_length=1)

class CityList(BaseModel):
    cities: list[CityLocation]

class MetricsResponse(BaseModel):
    metrics: list[dict[str, Any]]

@lru_cache(maxsize=1)
def _inference():
    return load_inference()

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/model/info")
def model_info() -> dict[str, Any]:
    try:
        return _inference().info()
    except InferenceError as exc:
        raise HTTPException(503, str(exc)) from exc

@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> dict[str, Any]:
    import csv, os
    from pathlib import Path
    root = Path(os.environ.get("AIRCAST_ROOT", Path(__file__).resolve().parents[1]))
    path = root / "artifacts" / "comparison.csv"
    if not path.is_file():
        raise HTTPException(503, f"Missing comparison artifact: {path.name}")
    with path.open(newline="", encoding="utf-8") as f:
        return {"metrics": list(csv.DictReader(f))}

@app.get("/cities", response_model=CityList)
def cities(q: str | None = Query(default=None, min_length=1)) -> CityList:
    try:
        if q:
            from aircast.providers import search_cities
            found = search_cities(q)
        else:
            found = curated_cities()
        return CityList(cities=found)
    except Exception as exc:
        raise HTTPException(503, f"City discovery unavailable: {exc}") from exc

@app.get("/air-quality", response_model=AirQualitySnapshot)
def air_quality(city_id: str = Query(min_length=1)) -> AirQualitySnapshot:
    city = next((c for c in curated_cities() if c.id == city_id), None)
    if city is None:
        raise HTTPException(404, f"Unknown or unsupported city: {city_id}")
    try:
        from aircast.providers import get_snapshot
        return get_snapshot(city)
    except Exception as exc:
        raise HTTPException(503, f"Current air-quality data unavailable: {exc}") from exc

@app.post("/predict", response_model=ForecastResult)
def predict(payload: PredictRequest) -> ForecastResult:
    try:
        from aircast.probabilistic import predict_city
        result = ForecastResult.model_validate(predict_city(payload.city_id))
        record_forecast(result)
        return result
    except (ImportError, InferenceError) as exc:
        raise HTTPException(503, f"Custom probabilistic forecast unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

@app.get("/monitoring", response_model=MonitoringReport)
def monitoring(min_samples: int = Query(default=30, ge=1, le=10000)) -> dict[str, Any]:
    return build_report(min_samples=min_samples)
