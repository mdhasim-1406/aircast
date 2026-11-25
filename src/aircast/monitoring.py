"""File-backed forecast/outcome ledger and delayed-outcome monitoring."""
from __future__ import annotations
import tempfile
import threading
import json, math, os
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import numpy as np


def _path() -> Path:
    root=Path(os.environ.get("AIRCAST_ROOT",Path(__file__).resolve().parents[2]))
    return Path(os.environ.get("AIRCAST_LEDGER",root/"data"/"platform-ledger.json"))

_WRITE_LOCK = threading.RLock()

@contextmanager
def _locked(path=None):
    p = Path(path) if path else _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(p.suffix + ".lock")
    with _WRITE_LOCK:
        with lock_path.open("a+b") as handle:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load(path=None):
    p=Path(path) if path else _path()
    try:
        data=json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data,dict): raise ValueError("ledger root must be an object")
        data.setdefault("forecasts",[]); data.setdefault("outcomes",[]); data.setdefault("references",{})
        return data
    except FileNotFoundError:
        return {"forecasts":[],"outcomes":[],"references":{}}

def _save(data,path=None):
    p=Path(path) if path else _path();p.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(data,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(temp,p)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def record_forecast(forecast:Any,path=None)->None:
    item=forecast.model_dump() if hasattr(forecast,"model_dump") else dict(forecast)
    with _locked(path):
        data=_load(path);key=(item["city_id"],item["forecast_date"],item["model_version"])
        if not any((r["city_id"],r["forecast_date"],r["model_version"])==key for r in data["forecasts"]):
            data["forecasts"].append({k:item.get(k) for k in ("city_id","forecast_date","predicted_pm25","p10","p90","model_version","issued_at")});_save(data,path)

def forecast_dates(city_id:str,path=None)->set[str]:
    with _locked(path):
        data=_load(path)
        return {r["forecast_date"] for r in data["forecasts"] if r["city_id"]==city_id}

def record_outcomes(frame:Any,provider="provider",path=None)->int:
    now=datetime.now(timezone.utc).isoformat();rows=frame.to_dict("records") if hasattr(frame,"to_dict") else frame
    with _locked(path):
        data=_load(path);by_key={(r["city_id"],r["observed_date"],r["provider"]):r for r in data["outcomes"]};added=0
        forecast_keys={(f["city_id"],f["forecast_date"]) for f in data["forecasts"]}
        for row in rows:
            value=row.get("pm25")
            if value is None or not math.isfinite(float(value)) or float(value)<0:continue
            city=str(row["city_id"]);observed=str(row.get("date",row.get("observed_date")))
            if (city,observed) not in forecast_keys:continue
            key=(city,observed,provider)
            if key not in by_key:added+=1
            by_key[key]={"city_id":city,"observed_date":observed,"pm25":float(value),"provider":provider,"recorded_at":now}
        data["outcomes"]=list(by_key.values());_save(data,path)
    return added

def set_reference(model_version:str,feature:str,values:list[float],path=None)->None:
    with _locked(path):
        data=_load(path)
        data["references"][f"{model_version}:{feature}"]={"model_version":model_version,"feature":feature,"values":values}
        _save(data,path)

def build_report(path=None,min_samples=30,freshness_hours=48):
    now=datetime.now(timezone.utc);data=_load(path);actuals={}
    for row in data["outcomes"]:actuals.setdefault((row["city_id"],row["observed_date"]),row)
    joined=[(f,actuals.get((f["city_id"],f["forecast_date"]))) for f in data["forecasts"]];observed=[(f,o) for f,o in joined if o is not None]
    errors=np.array([f["predicted_pm25"]-o["pm25"] for f,o in observed],dtype=float)
    perf={"status":"available" if len(observed)>=min_samples else "insufficient_data","sample_count":len(observed)}
    if len(observed)>=min_samples:
        perf.update(mae=float(np.mean(np.abs(errors))),bias=float(np.mean(errors)),rmse=float(np.sqrt(np.mean(errors**2))))
        covered=[(f,o) for f,o in observed if f.get("p10") is not None and f.get("p90") is not None]
        perf["interval_coverage"]=float(np.mean([f["p10"]<=o["pm25"]<=f["p90"] for f,o in covered])) if covered else None
        perf["interval_sample_count"]=len(covered)
    latest=max((datetime.fromisoformat(f["issued_at"].replace("Z","+00:00")) for f,_ in joined),default=None);age=(now-latest).total_seconds()/3600 if latest else None
    drift={r["feature"]:{"status":"insufficient_data","reason":"matching observed feature values are not recorded","reference_n":len(r["values"]),"current_n":0} for r in data["references"].values()} or {"status":"insufficient_data"}
    status="insufficient_data" if len(observed)<min_samples else "ok"
    if age is None or age>freshness_hours:status="stale" if status=="ok" else "insufficient_data"
    return {"generated_at":now.isoformat(),"status":status,"sample_count":len(joined),"outcome_sample_count":len(observed),"freshness":{"latest_forecast_age_hours":age,"max_age_hours":freshness_hours,"status":"unknown" if age is None else ("fresh" if age<=freshness_hours else "stale")},"drift":drift,"performance":perf}
