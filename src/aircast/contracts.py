"""Shared JSON contracts. Missing provider fields are null, never invented."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class CityLocation(BaseModel):
    id: str
    name: str
    state: str = ""
    country_code: str = "IN"
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    @property
    def label(self) -> str:
        return f"{self.name}, {self.state}" if self.state else self.name


class PollutantReading(BaseModel):
    value: float | None = None
    unit: str = "µg/m³"


class AirQualitySnapshot(BaseModel):
    city: CityLocation
    provider: str
    data_type: str
    timestamp: str
    retrieved_at: str
    pollutants: dict[str, PollutantReading] = Field(default_factory=dict)
    aqi: float | None = None
    aqi_standard: str | None = None
    station_id: str | None = None
    freshness: str = "unknown"


class ForecastResult(BaseModel):
    city_id: str
    forecast_date: str
    predicted_pm25: float = Field(ge=0)
    p10: float | None = Field(default=None, ge=0)
    p50: float | None = Field(default=None, ge=0)
    p90: float | None = Field(default=None, ge=0)
    unit: str = "µg/m³"
    model_name: str
    model_version: str
    data_cutoff: str
    issued_at: str
    provider: str = "AirCast"
    forecast_type: str = "custom_ml"


class DataQualityReport(BaseModel):
    dataset_version: str
    provider: str
    data_type: str
    cities: list[dict[str, Any]] = Field(default_factory=list)
    rules: dict[str, Any] = Field(default_factory=dict)
    generated_at: str


class ModelMetadata(BaseModel):
    version: str
    cities: list[str]
    dataset: dict[str, Any]
    cutoffs: dict[str, Any]
    features: list[str]


class MonitoringReport(BaseModel):
    generated_at: str
    status: str
    sample_count: int = 0
    model_version: str | None = None
    freshness: dict[str, Any] = Field(default_factory=dict)
    drift: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
