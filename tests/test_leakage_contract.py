"""Synthetic-only tests for chronological and leakage invariants."""
import numpy as np
import pandas as pd
import pytest
from aircast.features import make_supervised, latest_features, chronological_split
from aircast.data import daily_from_hourly


def daily():
    return pd.DataFrame({'date': pd.date_range('2023-01-01', periods=100), 'pm25': np.arange(100, dtype=float)})


def test_features_use_only_present_and_past():
    source = daily()
    first, columns = make_supervised(source)
    changed = source.copy()
    changed.loc[changed.date > pd.Timestamp('2023-03-01'), 'pm25'] += 9999
    second, _ = make_supervised(changed)
    a = first.set_index('date').loc['2023-03-01', columns].astype(float)
    b = second.set_index('date').loc['2023-03-01', columns].astype(float)
    np.testing.assert_allclose(a, b)
    assert first.set_index('date').loc['2023-03-01', 'pm25'] == 60


def test_gap_does_not_turn_next_available_day_into_tomorrow():
    source = daily().drop(index=50)
    rows, _ = make_supervised(source)
    assert all(rows.target_date == rows.date + pd.Timedelta(days=1))
    assert pd.Timestamp('2023-02-19') not in set(rows.date)


def test_latest_features_match_supervised_contract():
    source = daily()
    rows, columns = make_supervised(source)
    x, date = latest_features(source.iloc[:-1], columns)
    assert date == source.date.iloc[-2]
    np.testing.assert_allclose(x.iloc[0].astype(float), rows.iloc[-1][columns].astype(float))


def test_split_is_chronological():
    rows, _ = make_supervised(daily())
    train, validation, test = chronological_split(rows)
    assert train.target_date.max() <= validation.date.min()
    assert validation.target_date.max() <= test.date.min()


def test_duplicate_hours_do_not_fake_completeness():
    hourly = pd.DataFrame({'time': [pd.Timestamp('2023-01-01')]*24, 'pm2_5': [5.]*24})
    assert daily_from_hourly(hourly).empty


def test_insufficient_split_rejected():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({'x':[1,2]}))
