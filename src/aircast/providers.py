"""India-wide discovery and bounded, cached Open-Meteo access."""
from __future__ import annotations
import time
import threading
from datetime import date
import requests
import pandas as pd
from .contracts import CityLocation, AirQualitySnapshot, PollutantReading

CITIES = [
 ('chennai','Chennai','Tamil Nadu',13.0827,80.2707),('delhi','Delhi','Delhi',28.6139,77.209),
 ('mumbai','Mumbai','Maharashtra',19.076,72.8777),('bengaluru','Bengaluru','Karnataka',12.9716,77.5946),
 ('kolkata','Kolkata','West Bengal',22.5726,88.3639),('hyderabad','Hyderabad','Telangana',17.385,78.4867),
 ('pune','Pune','Maharashtra',18.5204,73.8567),('ahmedabad','Ahmedabad','Gujarat',23.0225,72.5714),
 ('lucknow','Lucknow','Uttar Pradesh',26.8467,80.9462),('jaipur','Jaipur','Rajasthan',26.9124,75.7873),
 ('kochi','Kochi','Kerala',9.9312,76.2673),('coimbatore','Coimbatore','Tamil Nadu',11.0168,76.9558),
 ('madurai','Madurai','Tamil Nadu',9.9252,78.1198),('patna','Patna','Bihar',25.5941,85.1376),
 ('kanpur','Kanpur','Uttar Pradesh',26.4499,80.3319),('bhopal','Bhopal','Madhya Pradesh',23.2599,77.4126),
 ('guwahati','Guwahati','Assam',26.1445,91.7362),('chandigarh','Chandigarh','Chandigarh',30.7333,76.7794),
 ('surat','Surat','Gujarat',21.1702,72.8311),('nagpur','Nagpur','Maharashtra',21.1458,79.0882),
 ('indore','Indore','Madhya Pradesh',22.7196,75.8577),('ranchi','Ranchi','Jharkhand',23.3441,85.3096),
 ('amritsar','Amritsar','Punjab',31.634,74.8723),('ludhiana','Ludhiana','Punjab',30.901,75.8573),
 ('agra','Agra','Uttar Pradesh',27.1767,78.0081),('varanasi','Varanasi','Uttar Pradesh',25.3176,82.9739),
 ('prayagraj','Prayagraj','Uttar Pradesh',25.4358,81.8463),('noida','Noida','Uttar Pradesh',28.5355,77.391),
 ('gurugram','Gurugram','Haryana',28.4595,77.0266),('thiruvananthapuram','Thiruvananthapuram','Kerala',8.5241,76.9366),
 ('kozhikode','Kozhikode','Kerala',11.2588,75.7804),('tiruchirappalli','Tiruchirappalli','Tamil Nadu',10.7905,78.7047),
 ('salem','Salem','Tamil Nadu',11.6643,78.146),('tirunelveli','Tirunelveli','Tamil Nadu',8.7139,77.7567),
 ('erode','Erode','Tamil Nadu',11.341,77.7172),('tiruppur','Tiruppur','Tamil Nadu',11.1085,77.3411),
 ('vellore','Vellore','Tamil Nadu',12.9165,79.1325),('visakhapatnam','Visakhapatnam','Andhra Pradesh',17.6868,83.2185),
 ('vijayawada','Vijayawada','Andhra Pradesh',16.5062,80.648),('tirupati','Tirupati','Andhra Pradesh',13.6288,79.4192),
 ('mysuru','Mysuru','Karnataka',12.2958,76.6394),('mangaluru','Mangaluru','Karnataka',12.9141,74.856),
 ('hubballi','Hubballi','Karnataka',15.3647,75.124),('bhubaneswar','Bhubaneswar','Odisha',20.2961,85.8245),
 ('cuttack','Cuttack','Odisha',20.4625,85.883),('shillong','Shillong','Meghalaya',25.5788,91.8933),
 ('dehradun','Dehradun','Uttarakhand',30.3165,78.0322),('shimla','Shimla','Himachal Pradesh',31.1048,77.1734),
 ('srinagar','Srinagar','Jammu and Kashmir',34.0837,74.7973),('jammu','Jammu','Jammu and Kashmir',32.7266,74.857),
 ('raipur','Raipur','Chhattisgarh',21.2514,81.6296),('panaji','Panaji','Goa',15.4909,73.8278),
 ('imphal','Imphal','Manipur',24.817,93.9368),('agartala','Agartala','Tripura',23.8315,91.2868)]
BASE='https://air-quality-api.open-meteo.com/v1/air-quality'
_CACHE={}
_LOCK=threading.Lock()

class ProviderError(RuntimeError):
    pass


def clear_cache():
    with _LOCK:
        _CACHE.clear()


def curated_cities():
    return [CityLocation(id=i,name=n,state=s,latitude=a,longitude=b) for i,n,s,a,b in CITIES]


def _get(url,params):
    key=(url,tuple(sorted(params.items())))
    with _LOCK:
        item=_CACHE.get(key)
    if item and time.monotonic()-item[0]<900:
        return item[1]
    error=None
    for attempt in range(2):
        try:
            r=requests.get(url,params=params,timeout=(4,18));r.raise_for_status();payload=r.json()
            payload['_retrieved_at']=pd.Timestamp.now(tz='UTC').isoformat()
            with _LOCK:
                if len(_CACHE)>256:_CACHE.clear()
                _CACHE[key]=(time.monotonic(),payload)
            return payload
        except (requests.RequestException,ValueError) as exc:
            error=exc
            if attempt==0:time.sleep(.5)
    raise ProviderError(f'Provider unavailable: {type(error).__name__}') from error


def search_cities(query):
    parts=[p.strip() for p in query.split(',') if p.strip()]
    if not parts:return []
    name=parts[0];state=parts[1].casefold() if len(parts)>1 else None
    known=[c for c in curated_cities() if name.casefold() in c.name.casefold() and (not state or state in c.state.casefold())]
    state_matches=[c for c in curated_cities() if name.casefold() == c.state.casefold()]
    if state_matches and not state:
        return state_matches
    payload=_get('https://geocoding-api.open-meteo.com/v1/search',{'name':name,'count':30,'language':'en','format':'json','countryCode':'IN'})
    found=list(known)
    for x in payload.get('results',[]):
        if x.get('country_code')!='IN' or (state and state not in x.get('admin1','').casefold()):continue
        if any(abs(c.latitude-x['latitude'])<.15 and abs(c.longitude-x['longitude'])<.15 for c in found):continue
        found.append(CityLocation(id=f"geo-{x['id']}",name=x['name'],state=x.get('admin1',''),latitude=x['latitude'],longitude=x['longitude']))
    return found


def get_hourly(city,past_days=7,forecast_days=3):
    if not 0<=past_days<=92 or not 1<=max(1,forecast_days)<=7:raise ValueError('Invalid provider window')
    payload=_get(BASE,{'latitude':city.latitude,'longitude':city.longitude,'hourly':'pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,us_aqi','past_days':past_days,'forecast_days':max(1,forecast_days),'timezone':'Asia/Kolkata'})
    h=payload.get('hourly',{});times=h.get('time',[])
    frame=pd.DataFrame({'time':pd.to_datetime(times)})
    for original,col in [('pm2_5','pm25'),('pm10','pm10'),('nitrogen_dioxide','nitrogen_dioxide'),('sulphur_dioxide','sulphur_dioxide'),('carbon_monoxide','carbon_monoxide'),('ozone','ozone'),('us_aqi','us_aqi')]:
        frame[col]=h.get(original,[None]*len(times))
    frame.attrs.update(provider='Open-Meteo CAMS',data_type='model_estimate',retrieved_at=payload.get('_retrieved_at'),units=payload.get('hourly_units',{}),latitude=payload.get('latitude'),longitude=payload.get('longitude'))
    return frame


def get_snapshot(city):
    df=get_hourly(city,1,1);now=pd.Timestamp.now(tz='Asia/Kolkata')
    eligible=df.loc[df.time<=now.tz_localize(None)]
    if eligible.empty:raise ProviderError('No current or past provider estimate available')
    row=eligible.iloc[-1];ts=pd.Timestamp(row.time).tz_localize('Asia/Kolkata')
    names={'pm25':'pm2_5','pm10':'pm10','nitrogen_dioxide':'nitrogen_dioxide','sulphur_dioxide':'sulphur_dioxide','carbon_monoxide':'carbon_monoxide','ozone':'ozone'}
    values={key:PollutantReading(value=None if pd.isna(row[col]) else float(row[col])) for col,key in names.items()}
    return AirQualitySnapshot(city=city,provider='Open-Meteo CAMS',data_type='model_estimate',timestamp=ts.isoformat(),retrieved_at=df.attrs['retrieved_at'],pollutants=values,aqi=None if pd.isna(row.us_aqi) else float(row.us_aqi),aqi_standard='US AQI',freshness='fresh' if (now-ts).total_seconds()<=10800 else 'stale')


def fetch_history(city,start_date,end_date):
    params={'latitude':city.latitude,'longitude':city.longitude,'start_date':str(start_date),'end_date':str(end_date),'hourly':'pm2_5,pm10','timezone':'Asia/Kolkata'}
    air=_get(BASE,params)
    weather=_get('https://archive-api.open-meteo.com/v1/archive',{'latitude':city.latitude,'longitude':city.longitude,'start_date':str(start_date),'end_date':str(end_date),'daily':'temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean','timezone':'Asia/Kolkata','models':'era5'})
    frame=pd.DataFrame(air['hourly']).drop_duplicates('time');frame['date']=pd.to_datetime(frame.time).dt.date
    daily=frame.groupby('date').agg(pm25=('pm2_5','mean'),pm10=('pm10','mean'),hours=('pm2_5','count')).reset_index()
    daily=daily.loc[daily.hours>=18]
    w=pd.DataFrame(weather['daily']).rename(columns={'time':'date'});w['date']=pd.to_datetime(w.date).dt.date
    return daily.merge(w,on='date',how='left')
