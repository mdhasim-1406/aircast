"""Artifact-backed inference shared by API and dashboard; no MLflow dependency."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class InferenceError(RuntimeError):
    """Artifacts are missing or violate the published contract."""


@dataclass
class Inference:
    bundle: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def model(self) -> Any:
        return self.bundle["model"]

    def info(self) -> dict[str, Any]:
        return {key: self.metadata.get(key, self.bundle.get(key)) for key in ("model_name", "version", "data_cutoff", "forecast_date")}

    def predict(self) -> dict[str, Any]:
        features = self.metadata.get("latest_features")
        columns = self.bundle.get("feature_columns")
        if not isinstance(features, dict) or not isinstance(columns, list) or not columns:
            raise InferenceError("metadata latest_features and model feature_columns are required")
        missing = [column for column in columns if column not in features]
        if missing:
            raise InferenceError(f"latest_features missing required columns: {missing}")
        try:
            values = np.asarray([[float(features[column]) for column in columns]], dtype=float)
            prediction = float(np.asarray(self.model.predict(values)).reshape(-1)[0])
        except (TypeError, ValueError, IndexError, AttributeError) as exc:
            raise InferenceError(f"unable to predict from saved features: {exc}") from exc
        if not np.isfinite(prediction):
            raise InferenceError("model returned a non-finite prediction")
        return {"forecast_date": self.metadata.get("forecast_date"), "predicted_pm25": prediction, "unit": "µg/m³", "model_name": self.metadata.get("model_name", self.bundle.get("model_name")), "model_version": self.metadata.get("version", self.bundle.get("version")), "data_cutoff": self.metadata.get("data_cutoff")}


def load_inference(artifacts_dir: str | Path | None = None) -> Inference:
    root = Path(os.environ.get("AIRCAST_ROOT", Path(__file__).resolve().parents[2]))
    directory = Path(artifacts_dir) if artifacts_dir is not None else root / "artifacts"
    model_path, metadata_path = directory / "model.joblib", directory / "metadata.json"
    if not model_path.is_file():
        raise InferenceError(f"missing required artifact: {model_path}")
    if not metadata_path.is_file():
        raise InferenceError(f"missing required artifact: {metadata_path}")
    try:
        bundle = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise InferenceError(f"cannot load model artifacts: {exc}") from exc
    required = {"model", "model_name", "version", "feature_columns"}
    if not isinstance(bundle, dict) or required - bundle.keys():
        raise InferenceError(f"model bundle must contain {sorted(required)}")
    if not isinstance(metadata, dict):
        raise InferenceError("metadata.json must contain a JSON object")
    return Inference(bundle, metadata)
