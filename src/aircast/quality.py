"""Pandera validation and transparent daily coverage diagnostics."""
from __future__ import annotations
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors, SchemaError

SCHEMA=pa.DataFrameSchema({'city_id':pa.Column(str,nullable=False),'date':pa.Column(pa.DateTime,nullable=False),'pm25':pa.Column(float,nullable=True,checks=pa.Check.ge(0)),'pm10':pa.Column(float,nullable=True,checks=pa.Check.ge(0),required=False),'hours':pa.Column(int,nullable=False,checks=pa.Check.in_range(0,24))},checks=[pa.Check(lambda d:not d.duplicated(['city_id','date']).any(),error='duplicate city/date')],strict=False,coerce=True)


def validate_daily(frame):
    errors=[]
    try:SCHEMA.validate(frame,lazy=True)
    except (SchemaErrors,SchemaError) as exc:errors=[str(exc)]
    cities=[]
    for cid,g in frame.groupby('city_id'):
        dates=pd.to_datetime(g.date);expected=(dates.max()-dates.min()).days+1
        series=g.sort_values('date').pm25
        flat=series.eq(series.shift()).rolling(7,min_periods=7).sum().ge(7)
        cities.append({'city_id':str(cid),'rows':len(g),'start':str(dates.min().date()),'end':str(dates.max().date()),'missing_pm25_pct':float(series.isna().mean()*100),'missing_calendar_days':max(0,expected-dates.nunique()),'duplicate_days':int(dates.duplicated().sum()),'negative_readings':int(series.lt(0).sum()),'incomplete_days':int(g.hours.lt(18).sum()),'flatline_flags':int(flat.sum()),'station_count':0,'station_availability':'not available: model-derived grid','units':'µg/m³','completeness_pct':float(g.hours.sum()/(24*expected)*100),'missingness':{c:float(g[c].isna().mean()*100) for c in g.columns}})
    return {'valid':not errors,'rows':len(frame),'errors':errors,'columns':list(frame.columns),'cities':cities,'rules':{'pandera_schema':'pass' if not errors else 'fail','daily_minimum_hours':18,'flatline_rule':'seven identical successive daily means, flagged not removed','high_values':'retained; no pollution-peak clipping','target_imputation':'never','unit':'µg/m³','ground_station_coverage':'unavailable, CAMS grid estimates only'},'generated_at':pd.Timestamp.now(tz='UTC').isoformat(),'data_type':'model_estimate','provider':'Open-Meteo CAMS'}
