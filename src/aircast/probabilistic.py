"""Multicity quantile forecasting with chronological conformal calibration."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from .contracts import ForecastResult
from .features import make_supervised, chronological_split, latest_features

ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=ROOT/'artifacts'/'probabilistic'
QUANTILES=(.1,.5,.9)

def _ordered_quantiles(p10,p50,p90):
    values=np.sort(np.vstack([np.asarray(p10,float),np.asarray(p50,float),np.asarray(p90,float)]),axis=0)
    return values[0],values[1],values[2]

def make_city_features(daily:pd.DataFrame):
    from .features import make_supervised as _make
    frame,cols=_make(daily)
    return frame,cols

def _read_data(path=None):
    p=Path(path) if path else ROOT/'data'/'multicity_daily.parquet'
    if not p.is_file(): raise FileNotFoundError(f"real multicity training data unavailable: {p}")
    df=pd.read_parquet(p)
    required={'city_id','date','pm25'}
    if required-set(df): raise ValueError(f"Dataset missing columns: {sorted(required-set(df))}")
    df=df.copy(); df['date']=pd.to_datetime(df.date).dt.tz_localize(None)
    if df.pm25.isna().any(): raise ValueError('PM2.5 target contains nulls; target imputation is prohibited')
    return df,p

def _feature_frame(city):
    f,cols=make_city_features(city)
    if len(f)<60: raise ValueError(f"Insufficient valid chronology ({len(f)} rows)")
    return f,cols

def _models():
    return {'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
      'random_forest':RandomForestRegressor(n_estimators=160,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=1),
      **{f'lightgbm_q{int(q*100)}':LGBMRegressor(objective='quantile',alpha=q,n_estimators=180,learning_rate=.04,num_leaves=15,max_depth=5,min_child_samples=12,reg_lambda=1.0,verbosity=-1,n_jobs=1,random_state=42) for q in QUANTILES}}

def train(data_path=None, artifact_dir=None, experiment_name='aircast-probabilistic'):
    df,path=_read_data(data_path); out=Path(artifact_dir) if artifact_dir else ARTIFACTS; out.mkdir(parents=True,exist_ok=True)
    version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); staging=out/f'.{version}.staging'; staging.mkdir()
    predictions=[]; cover=[]; importance=[]; backtest=[]; city_models={}; cutoffs={}; metrics={}
    try:
      import mlflow
      mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', f'sqlite:///{ROOT / "mlflow.db"}'))
      mlflow.set_experiment(experiment_name)
      runctx=mlflow.start_run(run_name=f'multicity-{version}')
      run=runctx.__enter__()
      mlflow.log_param('dataset',str(path)); mlflow.log_param('cities',','.join(sorted(df.city_id.unique())))
      for city_id,g in df.groupby('city_id',sort=True):
        feat,cols=_feature_frame(g[['date','pm25']])
        # Fixed absolute date boundaries synchronize cities and prevent differing calendars shifting cutoffs.
        train_end=pd.Timestamp('2025-05-06'); validation_end=pd.Timestamp('2025-12-03'); calibration_end=pd.Timestamp('2026-04-22'); test_end=pd.Timestamp(pd.to_datetime(df.date).max().normalize())
        tr=feat[feat.target_date<=train_end]; va=feat[(feat.target_date>train_end)&(feat.target_date<=validation_end)]
        cal=feat[(feat.target_date>validation_end)&(feat.target_date<=calibration_end)]; te=feat[(feat.target_date>calibration_end)&(feat.target_date<=test_end)]
        if min(len(tr),len(va),len(cal),len(te))<5: raise ValueError(f'{city_id}: insufficient rows under synchronized cutoffs')
        a=len(tr); b=a+len(va)
        cutoffs[city_id]={'train_through':str(tr.target_date.max().date()),'validation_through':str(va.target_date.max().date()),'calibration_through':str(cal.target_date.max().date()),'test_from':str(te.target_date.min().date()),'test_through':str(te.target_date.max().date())}
        fitted={}; quant={}
        for name,model in _models().items():
          if name.startswith('lightgbm_q'):
            model.fit(tr[cols],tr.pm25); quant[name]=model
          else:
            model.fit(tr[cols],tr.pm25); fitted[name]=model
        # Validation selects from fitted estimators and baselines; baseline indexes reflect the same season one year earlier.
        candidates={'persistence':va.pm25_lag_0.to_numpy(),'seasonal_naive':va.pm25_lag_6.to_numpy()}
        for name,m in fitted.items(): candidates[name]=m.predict(va[cols])
        choice=min(candidates,key=lambda x:mean_absolute_error(va.pm25,candidates[x]))
        base=fitted.get(choice)
        raw={q:quant[f'lightgbm_q{int(q*100)}'].predict(cal[cols]) for q in QUANTILES}
        p10,p50,p90=_ordered_quantiles(raw[.1],raw[.5],raw[.9])
        scores=np.maximum.reduce([p10-cal.pm25.to_numpy(),cal.pm25.to_numpy()-p90,np.zeros(len(cal))])
        # 80% one-sided nonconformity adjustment with finite-sample split-conformal rank.
        rank=min(len(scores),int(np.ceil((len(scores)+1)*.8)))
        radius=float(np.sort(scores)[rank-1])
        testraw={q:quant[f'lightgbm_q{int(q*100)}'].predict(te[cols]) for q in QUANTILES}
        lo,mid,hi=_ordered_quantiles(testraw[.1]-radius,testraw[.5],testraw[.9]+radius)
        if choice not in ('persistence','seasonal_naive'):
          import copy
          base=copy.deepcopy(fitted[choice]); base.fit(feat.iloc[:b][cols],feat.iloc[:b].pm25)
        def point_pred(frame):
          if choice=='persistence': return frame.pm25_lag_0.to_numpy()
          if choice=='seasonal_naive': return frame.pm25_lag_6.to_numpy()
          return base.predict(frame[cols])
        point=point_pred(te)
        y=te.pm25.to_numpy(); mae=float(mean_absolute_error(y,point)); rmse=float(np.sqrt(mean_squared_error(y,point)))
        seasonal=te.pm25_lag_6.to_numpy(); persistence=te.pm25_lag_0.to_numpy()
        city_metrics={'selected_point_model':choice,'mae':mae,'rmse':rmse,'r2':float(r2_score(y,point)),'train_mae':float(mean_absolute_error(tr.pm25,point_pred(tr))),'validation_mae':float(mean_absolute_error(va.pm25,point_pred(va))),'persistence_mae':float(mean_absolute_error(y,persistence)),'seasonal_naive_mae':float(mean_absolute_error(y,seasonal)),'pinball_p10':float(np.maximum(.1*(y-lo),-.9*(y-lo)).mean()),'pinball_p50':float(np.maximum(.5*(y-mid),-.5*(y-mid)).mean()),'pinball_p90':float(np.maximum(.9*(y-hi),-.1*(y-hi)).mean()),'coverage_80':float(((y>=lo)&(y<=hi)).mean()),'mean_interval_width':float(np.mean(hi-lo)),'conformal_radius':radius}
        metrics[city_id]=city_metrics
        city_models[city_id]={'features':cols,'point_name':choice,'point_model':base,'quantile_models':quant,'conformal_radius':radius,'data_cutoff':str(pd.to_datetime(g.date).max().date())}
        for rowidx,(_,r) in enumerate(te.iterrows()):
          predictions.append({'city_id':city_id,'date':str(r.target_date.date()),'actual':float(r.pm25),'predicted':float(point[rowidx]),'p10':float(lo[rowidx]),'p50':float(mid[rowidx]),'p90':float(hi[rowidx]),'model':choice})
        cover.append({'city_id':city_id,'coverage_80':city_metrics['coverage_80'],'nominal':.8,'mean_interval_width':city_metrics['mean_interval_width'],'calibration_radius':radius,'test_n':len(te)})
        for q,m in quant.items():
          for col,val in zip(cols,m.feature_importances_): importance.append({'city_id':city_id,'model':q,'feature':col,'importance':int(val)})
        for name,yp in [('persistence',persistence),('seasonal_naive',seasonal)]: backtest.append({'city_id':city_id,'model':name,'mae':float(mean_absolute_error(y,yp)),'rmse':float(np.sqrt(mean_squared_error(y,yp))),'n':len(y)})
        for name,m in fitted.items():
          yp=point if name==choice else m.predict(te[cols]); backtest.append({'city_id':city_id,'model':name,'mae':float(mean_absolute_error(y,yp)),'rmse':float(np.sqrt(mean_squared_error(y,yp))),'n':len(y)})
        for nm,m in quant.items():
          yp=m.predict(te[cols]); backtest.append({'city_id':city_id,'model':nm,'mae':float(mean_absolute_error(y,yp)),'rmse':float(np.sqrt(mean_squared_error(y,yp))),'n':len(y)})
        # Expanding-origin evaluation: one model fit for every quarterly cutoff, next 30 days evaluated.
        for origin in pd.date_range('2025-01-01','2026-01-01',freq='3MS'):
          history=feat[feat.target_date<=origin]; future=feat[(feat.target_date>origin)&(feat.target_date<=origin+pd.Timedelta(days=30))]
          if len(history)<90 or len(future)<10: continue
          cv=make_pipeline(StandardScaler(),Ridge(alpha=10.0)); cv.fit(history[cols],history.pm25)
          backtest.append({'city_id':city_id,'model':'rolling_ridge','origin':str(origin.date()),'mae':float(mean_absolute_error(future.pm25,cv.predict(future[cols]))),'rmse':float(np.sqrt(mean_squared_error(future.pm25,cv.predict(future[cols])))),'n':len(future)})
      bundle={'version':version,'cities':city_models,'dataset':str(path),'features':sorted({c for v in city_models.values() for c in v['features']})}
      latest={}
      for cid,g in df.groupby('city_id'):
        g=g[pd.to_datetime(g.date).dt.date < datetime.now(timezone.utc).date()]
        feat,cols=_feature_frame(g[['date','pm25']]); row,cut=latest_features(g[['date','pm25']],cols)
        latest[cid]={'features':row.iloc[0].to_dict(),'cutoff':str(cut.date()),'forecast_date':str((cut+pd.Timedelta(days=1)).date())}
      bundle['latest']=latest
      joblib.dump(bundle,staging/'bundle.joblib')
      pd.DataFrame(predictions).to_csv(staging/'predictions.csv',index=False); pd.DataFrame(cover).to_csv(staging/'coverage.csv',index=False); pd.DataFrame(importance).to_csv(staging/'feature_importance.csv',index=False); pd.DataFrame(backtest).to_csv(staging/'backtest.csv',index=False)
      comp=pd.DataFrame([{'city_id':c,**m} for c,m in metrics.items()]); comp.to_csv(staging/'comparison.csv',index=False)
      pd.DataFrame(predictions).to_csv(staging/'reference.parquet.tmp',index=False); pd.read_csv(staging/'reference.parquet.tmp').to_parquet(staging/'reference.parquet',index=False); (staging/'reference.parquet.tmp').unlink()
      summary={'version':version,'cities':sorted(city_models),'cutoffs':cutoffs,'metrics':metrics,'dataset':str(path),'mlflow_run_id':run.info.run_id,'features':bundle['features'],'generated_at':datetime.now(timezone.utc).isoformat(),'model_name':'LightGBM quantile + split conformal','freshness_days':3}
      (staging/'metadata.json').write_text(json.dumps(summary,indent=2,default=str))
      mlflow.log_metrics({f'{c}.{k}':v for c,m in metrics.items() for k,v in m.items() if isinstance(v,(int,float))})
      mlflow.log_artifacts(str(staging))
      # validate bundle/load/inference before atomic publication
      joblib.load(staging/'bundle.joblib')
      final=out/version; os.replace(staging,final)
      tmp=out/'.current.json.tmp'; tmp.write_text(json.dumps({'version':version,'directory':version},indent=2)); os.replace(tmp,out/'current.json')
      mlflow.end_run('FINISHED')
      return summary
    except Exception:
      try: mlflow.end_run('FAILED')
      except Exception: pass
      import shutil; shutil.rmtree(staging,ignore_errors=True)
      raise

def _bundle_path(directory=None):
    root=Path(directory) if directory else ARTIFACTS
    if (root/'current.json').exists():
      manifest=json.loads((root/'current.json').read_text()); path=root/manifest['directory']
    else: path=root
    if not (path/'bundle.joblib').is_file(): raise FileNotFoundError(f'No published probabilistic model at {root}')
    return path

def load_bundle(directory=None):
    p=_bundle_path(directory); bundle=joblib.load(p/'bundle.joblib'); bundle['_artifact_path']=str(p); return bundle

def supported_cities():
    try: return sorted(load_bundle()['cities'])
    except FileNotFoundError: return []

def predict_city(city_id,history=None):
    b=load_bundle()
    if city_id not in b['cities']: raise ValueError(f'Unsupported model city: {city_id}')
    model=b['cities'][city_id]
    if history is None:
      try:
        from .providers import curated_cities, get_hourly
        city=next(c for c in curated_cities() if c.id==city_id)
        hourly=get_hourly(city,past_days=16,forecast_days=0)
        hourly['date']=pd.to_datetime(hourly.time).dt.date
        # The latest partial day is discarded; hourly PM2.5 is provider/CAMS-derived, not a ground observation.
        history=hourly.groupby('date').agg(pm25=('pm25','mean'),hours=('pm25','count')).reset_index()
        history=history.dropna(subset=['pm25'])
        history=history[history.hours>=18]
      except Exception as exc:
        raise ValueError(f'Unable to fetch recent hourly provider PM2.5 history for {city_id}: {exc}') from exc
    today=pd.Timestamp.now(tz='Asia/Kolkata').date()
    history=history[pd.to_datetime(history.date).dt.date<today].copy()
    if history.empty: raise ValueError('No completed historical days available')
    features,cut=latest_features(history,model['features']); cutoff=pd.Timestamp(cut).date()
    max_history_date=pd.to_datetime(history.date).dt.date.max()
    if max_history_date!=cutoff: raise ValueError('Provider history contains no complete contiguous recent day')
    if (pd.Timestamp.now().date()-cutoff).days>2: raise ValueError(f'Stale model input for {city_id}: cutoff {cutoff}')
    x=features[model['features']]
    choice=model['point_name']; pred=float(x.pm25_lag_0.iloc[0] if choice=='persistence' else x.pm25_lag_6.iloc[0] if choice=='seasonal_naive' else model['point_model'].predict(x)[0])
    raw=[float(model['quantile_models'][f'lightgbm_q{q}'].predict(x)[0]) for q in (10,50,90)]
    p10,p50,p90=_ordered_quantiles(raw[0]-model['conformal_radius'],raw[1],raw[2]+model['conformal_radius'])
    p10=max(0.,float(p10[0])); p50=max(p10,float(p50[0])); p90=max(p50,float(p90[0]))
    now=datetime.now(timezone.utc).isoformat()
    return ForecastResult(city_id=city_id,forecast_date=(pd.Timestamp(cutoff)+pd.Timedelta(days=1)).date().isoformat(),predicted_pm25=max(0.,pred),p10=p10,p50=p50,p90=p90,model_name='LightGBM quantile + split conformal',model_version=b['version'],data_cutoff=str(cutoff),issued_at=now).model_dump()
