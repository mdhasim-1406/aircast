"""Explicit daily forecast generation and outcome refresh, never model promotion."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aircast.probabilistic import supported_cities,predict_city
from aircast.monitoring import record_forecast,build_report
from scripts.refresh import refresh

def main():
    outcomes=refresh()
    failures=[]
    for city in supported_cities():
        try:record_forecast(predict_city(city))
        except Exception as exc:failures.append({'city':city,'error':str(exc)})
    report=build_report();report['refresh']=outcomes;report['forecast_failures']=failures
    path=ROOT/'artifacts/platform/monitoring.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(path)
    print(json.dumps(report,indent=2))
    return bool(failures or outcomes['failures'])

if __name__=='__main__':raise SystemExit(main())
