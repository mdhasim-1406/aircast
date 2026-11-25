from datetime import datetime
from types import SimpleNamespace
import pandas as pd
from zoneinfo import ZoneInfo


def test_refresh_uses_completed_india_day_at_utc_midnight_boundary(monkeypatch):
    from scripts import refresh as module
    from aircast import monitoring, providers
    class Clock:
        @staticmethod
        def now(tz):
            assert tz == ZoneInfo('Asia/Kolkata')
            return datetime(2026, 11, 26, 0, 15, tzinfo=tz)
    monkeypatch.setattr(module, 'datetime', Clock)
    monkeypatch.setattr(providers, 'curated_cities', lambda: [SimpleNamespace(id='chennai')])
    monkeypatch.setattr(monitoring, 'forecast_dates', lambda _: ['2026-11-25', '2026-11-26'])
    calls = []
    def history(city, start, end):
        calls.append((str(start), str(end)))
        return pd.DataFrame({'date': ['2026-11-25'], 'pm25': [12.0]})
    monkeypatch.setattr(providers, 'fetch_history', history)
    monkeypatch.setattr(monitoring, 'record_outcomes', lambda frame, provider: len(frame))
    assert module.refresh()['updated_outcomes'] == 1
    assert calls == [('2026-11-25', '2026-11-25')]
