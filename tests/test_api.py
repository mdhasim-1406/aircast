import json
from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient

from aircast.inference import InferenceError, load_inference
from api.main import app


def test_inference_predicts_from_bundled_contract(tmp_path):
    joblib.dump({"model": DummyModel(), "model_name": "test", "version": "v1", "feature_columns": ["pm25"], "metadata": {"data_cutoff": "2024-01-01", "forecast_date": "2024-01-02", "latest_features": {"pm25": 12.0}}}, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"model_name": "test", "version": "v1", "data_cutoff": "2024-01-01", "forecast_date": "2024-01-02", "latest_features": {"pm25": 12.0}}))
    result = load_inference(tmp_path).predict()
    assert result["predicted_pm25"] == 13.0
    assert result["forecast_date"] == "2024-01-02"


def test_api_health_and_info():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model/info").status_code in (200, 503)


def test_missing_artifacts_fail_explicitly(tmp_path):
    try:
        load_inference(tmp_path)
    except InferenceError as exc:
        assert "model.joblib" in str(exc)
    else:
        raise AssertionError("missing model should fail")


class DummyModel:
    def predict(self, values):
        return np.array([values[0][0] + 1])
