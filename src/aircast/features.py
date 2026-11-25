"""Strictly past-only daily features for next-day PM2.5 forecasts."""
from __future__ import annotations
import numpy as np
import pandas as pd
TARGET = "pm25"

class PersistenceRegressor:
    """Sklearn-compatible persistence model with no MLflow dependency."""
    def fit(self, X, y=None): return self
    def predict(self, X):
        return np.asarray(X["pm25_lag_0"] if hasattr(X,"columns") else X[:,0],dtype=float)

def make_supervised(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame=daily.copy(); frame["date"]=pd.to_datetime(frame.date)
    if "pm25" not in frame and "pm2_5" in frame: frame=frame.rename(columns={"pm2_5":"pm25"})
    frame=frame.sort_values("date").drop_duplicates("date",keep="last").set_index("date")
    if frame.index.isna().any() or frame.pm25.isna().any(): raise ValueError("Daily data requires valid dates and PM2.5")
    calendar=pd.date_range(frame.index.min(),frame.index.max(),freq="D",name="date")
    frame=frame.reindex(calendar); frame.index.name="date"
    features=[]
    for lag in (0,1,2,3,6,7,14):
        name=f"pm25_lag_{lag}"; frame[name]=frame.pm25.shift(lag); features.append(name)
    for window in (3,7,14):
        name=f"pm25_roll_mean_{window}"; frame[name]=frame.pm25.shift(1).rolling(window,min_periods=window).mean(); features.append(name)
        name=f"pm25_roll_std_{window}"; frame[name]=frame.pm25.shift(1).rolling(window,min_periods=window).std(); features.append(name)
    frame["day_of_week_sin"]=np.sin(2*np.pi*frame.index.dayofweek/7); frame["day_of_week_cos"]=np.cos(2*np.pi*frame.index.dayofweek/7)
    frame["day_of_year_sin"]=np.sin(2*np.pi*frame.index.dayofyear/365.25); frame["day_of_year_cos"]=np.cos(2*np.pi*frame.index.dayofyear/365.25)
    features += ["day_of_week_sin","day_of_week_cos","day_of_year_sin","day_of_year_cos"]
    frame["date"]=frame.index
    frame["target_date"]=frame.index.to_series().shift(-1).to_numpy()
    frame[TARGET]=frame.pm25.shift(-1)
    frame=frame.loc[frame.target_date.eq(frame.date+pd.Timedelta(days=1))].dropna(subset=features+[TARGET]).reset_index(drop=True)
    return frame,features

def latest_features(daily: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compute inference row for latest completed day, without requiring tomorrow's label."""
    frame=daily.copy(); frame["date"]=pd.to_datetime(frame.date)
    if "pm25" not in frame and "pm2_5" in frame: frame=frame.rename(columns={"pm2_5":"pm25"})
    frame=frame.sort_values("date").drop_duplicates("date",keep="last").set_index("date")
    idx=pd.date_range(frame.index.min(),frame.index.max(),freq="D")
    frame=frame.reindex(idx); frame.index.name="date"
    x={}
    for col in columns:
        if col.startswith("pm25_lag_"):
            lag=int(col.rsplit("_",1)[1]); x[col]=frame.pm25.iloc[-1-lag]
        elif col.startswith("pm25_roll_mean_") or col.startswith("pm25_roll_std_"):
            kind,window=col.replace("pm25_roll_","").split("_"); vals=frame.pm25.iloc[-int(window)-1:-1]
            x[col]=vals.mean() if kind=="mean" else vals.std()
        elif col=="day_of_week_sin": x[col]=np.sin(2*np.pi*frame.index[-1].dayofweek/7)
        elif col=="day_of_week_cos": x[col]=np.cos(2*np.pi*frame.index[-1].dayofweek/7)
        elif col=="day_of_year_sin": x[col]=np.sin(2*np.pi*frame.index[-1].dayofyear/365.25)
        elif col=="day_of_year_cos": x[col]=np.cos(2*np.pi*frame.index[-1].dayofyear/365.25)
        else: raise ValueError(f"No latest feature implementation for {col}")
    result=pd.DataFrame([x],columns=columns)
    if result.isna().any(axis=None): raise ValueError("Insufficient contiguous history for latest forecast features")
    return result,frame.index[-1]

def chronological_split(frame: pd.DataFrame,train_fraction=.70,validation_fraction=.15):
    if not 0<train_fraction<1 or not 0<validation_fraction<1 or train_fraction+validation_fraction>=1: raise ValueError("Invalid split fractions")
    n=len(frame); a=int(n*train_fraction); b=int(n*(train_fraction+validation_fraction))
    if a<2 or b<=a or n<=b: raise ValueError(f"Not enough contiguous rows for train/validation/test (got {n})")
    return frame.iloc[:a].copy(),frame.iloc[a:b].copy(),frame.iloc[b:].copy()
