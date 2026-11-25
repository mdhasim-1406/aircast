"""Refresh provider history for already completed forecast dates only."""
from __future__ import annotations
import argparse, json
from datetime import date, timedelta


def refresh() -> dict:
    from aircast.monitoring import forecast_dates, record_outcomes
    from aircast.providers import curated_cities, fetch_history
    updated, failures = 0, []
    cutoff = date.today() - timedelta(days=1)
    for city in curated_cities():
        try:
            targets = forecast_dates(city.id)
            if not targets:
                continue
            start = min(date.fromisoformat(d) for d in targets)
            end = min(max(date.fromisoformat(d) for d in targets), cutoff)
            if start > end:
                continue
            frame = fetch_history(city, start, end).copy()
            frame["city_id"] = city.id
            frame = frame[frame["date"].astype(str).isin(targets)]
            updated += record_outcomes(frame, provider="aircast_provider")
        except Exception as exc:
            failures.append({"city_id": city.id, "error": str(exc)})
    return {"updated_outcomes": updated, "failures": failures, "model_promotion": "not_performed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("explicit --run required")
    result = refresh()
    print(json.dumps(result, sort_keys=True))
    return 1 if result["failures"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
