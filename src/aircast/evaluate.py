"""Evaluate the saved model and write independently auditable metrics."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import load_daily, project_root
from .features import make_supervised, chronological_split
from .train import metric

def evaluate(daily: pd.DataFrame | None=None):
    root=project_root(); art=root/"artifacts"
    bundle=joblib.load(art/"model.joblib")
    daily=load_daily() if daily is None else daily.copy()
    supervised,columns=make_supervised(daily)
    if columns!=bundle["feature_columns"]: raise ValueError("Saved model feature schema does not match current data")
    tr,va,te=chronological_split(supervised)
    X=te[columns]; actual=te.pm25.to_numpy(); predicted=np.asarray(bundle["model"].predict(X)); baseline=te.pm25_lag_0.to_numpy()
    result={"model_name":bundle["model_name"],"version":bundle["version"],"split":"test","metrics":metric(actual,predicted),"persistence_metrics":metric(actual,baseline),"test_start":te.target_date.min().strftime("%Y-%m-%d"),"test_end":te.target_date.max().strftime("%Y-%m-%d"),"rows":int(len(te))}
    with open(art/"evaluation.json","w",encoding="utf8") as f: json.dump(result,f,indent=2,allow_nan=False)
    return result

if __name__=="__main__": print(json.dumps(evaluate(),indent=2))
