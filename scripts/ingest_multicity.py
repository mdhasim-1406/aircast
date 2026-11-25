#!/usr/bin/env python3
"""Acquire multicity historical archive; retain source/method provenance."""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aircast.providers import curated_cities, fetch_history
from aircast.quality import validate_daily

def main():
    end=date.today()-timedelta(days=6) # ERA5 archive publication lag
    start=date(2023,1,1)
    cities=[c for c in curated_cities() if c.id in {'chennai','delhi','mumbai','bengaluru','kolkata'}]
    frames=[]; errors=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs={pool.submit(fetch_history,c,start,end):c for c in cities}
        for f in as_completed(jobs):
            c=jobs[f]
            try:
                df=f.result(); df.insert(0,'city_id',c.id); frames.append(df)
                print(f'{c.id}: {len(df)} daily rows, {df.date.min()}..{df.date.max()}')
            except Exception as e:
                errors.append({'city_id':c.id,'error':str(e)}); print(f'{c.id}: ERROR {e}',file=sys.stderr)
    if not frames: raise SystemExit('No city history downloaded')
    data=pd.concat(frames,ignore_index=True); data['date']=pd.to_datetime(data.date)
    result=validate_daily(data)
    if not result['valid']: raise SystemExit(json.dumps(result,indent=2))
    out=ROOT/'data/multicity_daily.parquet'; out.parent.mkdir(parents=True,exist_ok=True); data.to_parquet(out,index=False)
    prov={'dataset':'multicity_daily.parquet','date_range':{'start':str(start),'end':str(end)},'cities_requested':[c.id for c in cities],'cities_succeeded':sorted(data.city_id.unique().tolist()),'errors':errors,'row_count':len(data),'columns':list(data.columns),'sources':{'pollutants':{'provider':'Open-Meteo Air Quality API','endpoint':'https://air-quality-api.open-meteo.com/v1/air-quality','status':'model-derived atmospheric-composition forecasts; not ground-station measurements'},'weather':{'provider':'Copernicus ERA5 via Open-Meteo Historical Weather API','endpoint':'https://archive-api.open-meteo.com/v1/archive','model':'era5','status':'reanalysis; estimated weather covariates'}},'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(),'validation':result}
    (ROOT/'artifacts/platform').mkdir(parents=True,exist_ok=True)
    (ROOT/'artifacts/platform/provenance.json').write_text(json.dumps(prov,indent=2))
    (ROOT/'artifacts/platform/quality.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'rows':len(data),'cities':data.city_id.nunique(),'quality':result['valid'],'errors':errors},indent=2))
if __name__=='__main__': main()
