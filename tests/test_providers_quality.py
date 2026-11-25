"""Deterministic provider fixtures, never used as live or training data."""
import pandas as pd
import pytest
import requests
from pydantic import ValidationError
from aircast import providers as p
from aircast.contracts import CityLocation
from aircast.quality import validate_daily


def test_catalogue_and_state_search():
    cities=p.curated_cities()
    assert len(cities)>=50 and len({c.id for c in cities})==len(cities)
    assert cities[0].id=='chennai'
    assert {c.id for c in p.search_cities('Karnataka')} >= {'bengaluru','mysuru','hubballi'}


def test_ambiguous_search_and_india_filter(monkeypatch):
    monkeypatch.setattr(p,'_get',lambda *a,**k:{'results':[{'id':1,'name':'Test','admin1':'Punjab','country_code':'IN','latitude':30.,'longitude':75.},{'id':2,'name':'Test','admin1':'Other','country_code':'US','latitude':30.,'longitude':75.}]})
    found=p.search_cities('Test, Punjab')
    assert len(found)==1 and found[0].state=='Punjab'


def test_null_pollutants_and_current_not_future(monkeypatch):
    now=pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None).floor('h')
    df=pd.DataFrame({'time':[now-pd.Timedelta(hours=1),now+pd.Timedelta(hours=3)],'pm25':[12.,999.],'pm10':[None,None],'nitrogen_dioxide':[None,None],'sulphur_dioxide':[None,None],'carbon_monoxide':[None,None],'ozone':[None,None],'us_aqi':[42.,999.]})
    df.attrs['retrieved_at']='2026-01-01T00:00:00+00:00'
    monkeypatch.setattr(p,'get_hourly',lambda *a,**k:df)
    s=p.get_snapshot(p.curated_cities()[0])
    assert s.pollutants['pm2_5'].value==12 and s.pollutants['pm10'].value is None
    assert s.data_type=='model_estimate' and s.aqi_standard=='US AQI'
    assert s.retrieved_at==df.attrs['retrieved_at']


def test_timeout_is_explicit(monkeypatch):
    p.clear_cache()
    def fail(*a,**k):raise requests.Timeout('fixture')
    monkeypatch.setattr(p.requests,'get',fail);monkeypatch.setattr(p.time,'sleep',lambda _:None)
    with pytest.raises(p.ProviderError):p._get('https://fixture.invalid',{})


def test_invalid_coordinates():
    with pytest.raises(ValidationError):CityLocation(id='x',name='x',latitude=99,longitude=80)


def test_quality_rejects_negative_and_duplicates():
    row={'city_id':'delhi','date':pd.Timestamp('2024-01-01'),'pm25':4.,'pm10':8.,'hours':24}
    assert validate_daily(pd.DataFrame([row]))['valid']
    assert not validate_daily(pd.DataFrame([{**row,'pm25':-1}]))['valid']
    assert not validate_daily(pd.DataFrame([row,row]))['valid']
