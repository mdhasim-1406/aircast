import pandas as pd
import pytest

from aircast.probabilistic import load_bundle, predict_city, supported_cities


def test_inference_rejects_stale_history(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        predict_city("unknown", pd.DataFrame())


def test_quantiles_ordered_and_manifest_points_to_bundle(tmp_path):
    from aircast.probabilistic import _ordered_quantiles
    out = _ordered_quantiles([4.0], [2.0], [8.0])
    assert out[0][0] <= out[1][0] <= out[2][0]


def test_supervised_gap_does_not_bridge_missing_day():
    from aircast.probabilistic import make_city_features
    d = pd.DataFrame({"date":pd.to_datetime(["2024-01-01","2024-01-02","2024-01-04"]),"pm25":[1.,2.,4.]})
    X, _ = make_city_features(d)
    assert not ((X.date == pd.Timestamp("2024-01-02")).any())
