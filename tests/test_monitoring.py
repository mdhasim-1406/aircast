from datetime import date, timedelta

from aircast.monitoring import build_report, record_forecast, record_outcomes, set_reference


def test_outcome_join_and_insufficient_samples(tmp_path):
    db = tmp_path / "monitor.json"
    record_forecast({"city_id":"delhi","forecast_date":"2025-01-01","predicted_pm25":20,"model_version":"v1","issued_at":"2025-01-01T00:00:00+00:00"}, db)
    assert record_outcomes([{"city_id":"delhi","date":"2025-01-01","pm25":12}], path=db) == 1
    report = build_report(db, min_samples=2)
    assert report["outcome_sample_count"] == 1
    assert report["performance"]["status"] == "insufficient_data"


def test_metrics_and_drift_are_emitted_only_with_minimum_samples(tmp_path):
    db = tmp_path / "monitor.json"
    set_reference("v1", "pm25", list(range(1, 41)), db)
    for i in range(40):
        observed_date = (date(2025, 1, 1) + timedelta(days=i)).isoformat()
        issued = f"{observed_date}T00:00:00+00:00"
        record_forecast({"city_id":"delhi","forecast_date":observed_date,"predicted_pm25":float(i+11),"p10":float(i+5),"p90":float(i+17),"model_version":"v1","issued_at":issued}, db)
        record_outcomes([{"city_id":"delhi","date":observed_date,"pm25":float(i+10)}], path=db)
    report = build_report(db, min_samples=30)
    assert report["performance"]["mae"] == 1
    assert report["performance"]["interval_coverage"] == 1
    assert report["drift"]["pm25"]["status"] == "insufficient_data"
