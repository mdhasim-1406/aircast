"""CPU-bounded validation-selected next-day Chennai PM2.5 training."""
from __future__ import annotations
import json, os
import hashlib
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import load_daily, project_root
from .features import make_supervised, chronological_split, latest_features, PersistenceRegressor

def models():
    return {
      "persistence": PersistenceRegressor(),
      "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
      "random_forest": RandomForestRegressor(n_estimators=160,max_depth=8,min_samples_leaf=3,max_features=0.8,n_jobs=1,random_state=42),
      "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=1.0,random_state=42),
      "xgboost": XGBRegressor(n_estimators=120,max_depth=3,learning_rate=.04,subsample=.8,colsample_bytree=.8,reg_lambda=2,n_jobs=1,random_state=42,verbosity=0),
    }

def metric(y,p):
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p))}

def train(daily: pd.DataFrame | None=None):
    root=project_root(); daily=load_daily() if daily is None else daily.copy()
    supervised, columns=make_supervised(daily)
    tr,va,te=chronological_split(supervised)
    X=lambda f:f[columns]; y=lambda f:f["pm25"]
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",f"file:{root/'mlruns'}"))
    records=[]; val_predictions={}; fitted={}
    with mlflow.start_run(run_name="chennai-pm25-validation-selection") as run:
        mlflow.log_params({"location":"Chennai, India","source":"Open-Meteo CAMS","target":"next_day_daily_mean_pm25","train_rows":len(tr),"validation_rows":len(va),"test_rows":len(te),"features":len(columns)})
        for name,model in models().items():
            model.fit(X(tr),y(tr)); pred=np.asarray(model.predict(X(va)))
            val_predictions[name]=pred; fitted[name]=model
            m=metric(y(va),pred); records.append({"model":name,"split":"validation",**m})
            mlflow.log_metrics({f"validation_{name}_{k}":v for k,v in m.items()})
        # Tiny bounded expanding-window CV on training chronology only; final selection uses validation MAE.
        # Grid size is deliberately bounded to preserve CPU/time.
        cv=TimeSeriesSplit(n_splits=3)
        for name in ("ridge", "random_forest", "xgboost"):
            model=fitted[name]
            grid={"ridge__alpha":[1.0,10.0,50.0]} if name=="ridge" else ({"max_depth":[6,8],"min_samples_leaf":[3,5]} if name=="random_forest" else {"max_depth":[2,3],"n_estimators":[80,120]})
            gs=GridSearchCV(model,grid,scoring="neg_mean_absolute_error",cv=cv,n_jobs=1,refit=True)
            gs.fit(X(tr),y(tr)); fitted[name]=gs.best_estimator_
            val_predictions[name]=np.asarray(gs.best_estimator_.predict(X(va)))
            mlflow.log_params({f"{name}_{k}":v for k,v in gs.best_params_.items()})
        records=[r for r in records if r["split"]!="validation"]
        for name,pred in val_predictions.items():
            m=metric(y(va),pred); records.append({"model":name,"split":"validation",**m})
            mlflow.log_metrics({f"validation_{name}_{k}":v for k,v in m.items()})
        # Model selection is based only on validation MAE, with stable tie-break.
        winner=min(val_predictions,key=lambda name:(metric(y(va),val_predictions[name])["mae"],name))
        # Preserve selected tuned parameters, then refit only on train (not validation/test).
        chosen=fitted[winner]; chosen.fit(X(tr),y(tr))
        # Test is evaluated once, after selection; no fitting or tuning on test.
        prediction=np.asarray(chosen.predict(X(te)))
        baseline=np.asarray(fitted["persistence"].predict(X(te)))
        for name, model in fitted.items():
            if name==winner: model=chosen
            pm=np.asarray(model.predict(X(te)))
            records.append({"model":name,"split":"test",**metric(y(te),pm)})
        comparison=pd.DataFrame(records)
        preds=pd.DataFrame({"date":te.target_date.dt.strftime("%Y-%m-%d"),"actual":y(te).to_numpy(),"predicted":prediction,"baseline":baseline,"split":"test"})
        for split, subset in (("train",tr),("validation",va)):
            m=np.asarray(chosen.predict(X(subset)))
            preds=pd.concat([preds,pd.DataFrame({"date":subset.target_date.dt.strftime("%Y-%m-%d"),"actual":y(subset).to_numpy(),"predicted":m,"baseline":fitted['persistence'].predict(X(subset)),"split":split})],ignore_index=True)
        artifacts=root/"artifacts"; artifacts.mkdir(parents=True,exist_ok=True)
        comparison.to_csv(artifacts/"comparison.csv",index=False); preds.sort_values("date").to_csv(artifacts/"predictions.csv",index=False)
        daily_out=daily.copy(); daily_out["date"]=pd.to_datetime(daily_out.date).dt.strftime("%Y-%m-%d"); daily_out.to_csv(artifacts/"daily.csv",index=False)
        importance=[]
        if hasattr(chosen,"feature_importances_"): importance=[{"feature":f,"importance":float(v)} for f,v in zip(columns,chosen.feature_importances_)]
        elif hasattr(chosen,"coef_"): importance=[{"feature":f,"importance":float(abs(v))} for f,v in zip(columns,chosen.coef_)]
        if importance: pd.DataFrame(importance).sort_values("importance",ascending=False).to_csv(artifacts/"feature_importance.csv",index=False)
        latest_x,latest_date=latest_features(daily,columns)
        cutoff=pd.Timestamp(latest_date).strftime("%Y-%m-%d"); forecast=(pd.Timestamp(cutoff)+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        latest_row=latest_x.iloc[0]
        latest_feature_values={f:float(latest_row[f]) for f in columns}
        version=run.info.run_id[:12]
        data_hash=hashlib.sha256(daily.to_csv(index=False).encode()).hexdigest()
        metadata={"model_name":winner,"version":version,"data_cutoff":cutoff,"forecast_date":forecast,"latest_features":latest_feature_values,"dataset":{"source":"Open-Meteo CAMS model-derived air-quality API","location":"Chennai, India","latitude":13.0827,"longitude":80.2707,"daily_rows":int(len(daily)),"first_date":pd.Timestamp(daily.date.min()).strftime("%Y-%m-%d"),"last_date":cutoff,"training_rows":int(len(tr)),"validation_rows":int(len(va)),"test_rows":int(len(te)),"sha256":data_hash}}
        bundle={"model":chosen,"model_name":winner,"version":version,"feature_columns":columns,"metadata":metadata}
        joblib.dump(bundle,artifacts/"model.joblib")
        with open(artifacts/"metadata.json","w",encoding="utf8") as f: json.dump(metadata,f,indent=2,allow_nan=False)
        mlflow.log_artifacts(str(artifacts),artifact_path="artifacts")
        mlflow.log_param("selected_model",winner); mlflow.log_metric("test_mae",metric(y(te),prediction)["mae"])
    return {"model_name":winner,"metadata":metadata,"comparison":comparison,"predictions":preds}

if __name__=="__main__":
    result=train(); print(f"Selected {result['model_name']}; artifacts saved in {project_root()/'artifacts'}")
