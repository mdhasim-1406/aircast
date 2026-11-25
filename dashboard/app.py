from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(os.environ.get("AIRCAST_ROOT", Path(__file__).resolve().parents[1])).expanduser()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="AirCast | India air quality", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root{--bg:#0b1117;--panel:#111b24;--panel2:#14212b;--line:#24323d;--text:#e4edf1;--muted:#91a3ad;--teal:#58c7bd;--cyan:#83d9e8;--amber:#f0bd72;--red:#f18c83}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text)}.stApp{background:var(--bg)}
.block-container{max-width:1440px;padding:1.2rem 3.2rem 3rem}header[data-testid="stHeader"]{background:transparent}
.brand{display:flex;align-items:center;gap:11px;margin:0 0 25px}.brandmark{height:29px;width:29px;border:1px solid var(--teal);border-radius:50%;position:relative}.brandmark:before,.brandmark:after{content:'';position:absolute;height:1px;left:5px;right:5px;top:13px;background:var(--teal)}.brandmark:after{transform:rotate(-32deg)}.brandname{font-weight:700;letter-spacing:.08em;font-size:14px}.brandplace{color:var(--muted);font-size:11px;margin-left:12px}
.eyebrow,.label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.14em;font-weight:700}.hero{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:24px 0 27px;margin-bottom:19px;display:flex;justify-content:space-between;align-items:end;gap:20px}.hero h1{font:400 clamp(32px,4vw,52px)/1.03 Newsreader,serif;letter-spacing:-.8px;margin:9px 0;color:var(--text)}.hero p{color:var(--muted);margin:0;font-size:13px}.hero-right{border-left:1px solid var(--line);padding:4px 0 4px 22px;min-width:190px}.hero-num{font:400 38px DM Mono,monospace;color:var(--teal);line-height:1}.hero-sub{font-size:11px;color:var(--muted);margin-top:7px}
.section{font:500 24px Newsreader,serif;margin:23px 0 11px}.subtle{color:var(--muted);font-size:12px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:15px 17px;margin-bottom:12px}.reading{background:var(--panel);border:1px solid var(--line);padding:13px 14px;border-radius:4px;min-height:85px}.reading-name{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.1em}.reading-value{font:21px DM Mono,monospace;margin-top:9px}.reading-unit{font:11px 'DM Sans',sans-serif;color:var(--muted)}.stTabs [data-baseweb="tab-list"]{gap:22px;border-bottom:1px solid var(--line)}.stTabs [data-baseweb="tab"]{padding:10px 2px;font-size:13px}.stTabs [aria-selected="true"]{color:var(--teal)}.stButton>button{border:1px solid #31545a;background:#13262c;color:var(--text);border-radius:4px}.stButton>button:hover{border-color:var(--teal);color:var(--teal)}.stTextInput input,.stSelectbox [data-baseweb="select"]>div{background:#101922!important;border-color:var(--line)!important;color:var(--text)!important}.stDataFrame{border:1px solid var(--line)}[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);padding:11px 13px;border-radius:4px}.stAlert{background:var(--panel);border:1px solid var(--line)}.stCaption{color:var(--muted)}footer{visibility:hidden}
@media(max-width:760px){.block-container{padding:.7rem .85rem 2rem}.hero{align-items:flex-start;flex-direction:column;padding:17px 0}.hero-right{border-left:0;border-top:1px solid var(--line);padding:13px 0 0;width:100%}.brand{margin-bottom:15px}.section{margin-top:20px}.stTabs [data-baseweb="tab-list"]{gap:12px}.stTabs [data-baseweb="tab"]{font-size:12px;padding:8px 0}.reading{padding:10px;min-height:78px}.reading-value{font-size:18px}}
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def _load_json(path: str, mtime: float) -> dict:
    del mtime
    with open(path, encoding="utf-8") as f: return json.load(f)

@st.cache_data(ttl=300, show_spinner=False)
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def _provider_module():
    from aircast import providers
    return providers

def _artifact(path: str) -> Path: return ROOT / path

def _read_json(path: Path) -> dict | None:
    try: return _load_json(str(path), path.stat().st_mtime) if path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError): return None

def _read_csv(path: Path) -> pd.DataFrame | None:
    try: return _load_csv(str(path), path.stat().st_mtime) if path.is_file() else None
    except (OSError, ValueError, pd.errors.ParserError): return None

def _safe_call(module: Any, name: str, *args, **kwargs):
    try: return getattr(module, name)(*args, **kwargs), None
    except Exception as exc: return None, f"{type(exc).__name__}: {exc}"

def _city_label(city: Any) -> str:
    if hasattr(city, 'label'): return city.label
    if isinstance(city, dict): return f"{city.get('name','')}, {city.get('state','')}".strip(', ')
    return str(city)

def _city_id(city: Any) -> str:
    return str(city.get('id') if isinstance(city, dict) else getattr(city, 'id', ''))

def _display_city(city: Any) -> dict:
    if hasattr(city, 'model_dump'): return city.model_dump()
    return dict(city) if isinstance(city, dict) else {k:getattr(city,k,None) for k in ('id','name','state','country_code','latitude','longitude')}

def _section(title: str, subtitle: str = ''):
    st.markdown(f'<div class="section">{title}</div><div class="subtle">{subtitle}</div>', unsafe_allow_html=True)

def _quality_cards(report: dict):
    summary = report.get('summary', report)
    keys = [('dataset_version','Dataset version'),('provider','Source'),('data_type','Data type'),('generated_at','Report generated')]
    visible=[(label,summary.get(key)) for key,label in keys if summary.get(key) is not None]
    if visible:
        cols=st.columns(min(4,len(visible)))
        for col,(label,value) in zip(cols,visible): col.metric(label,str(value))
    cities=report.get('cities',[])
    if cities:
        frame=pd.DataFrame(cities)
        st.markdown('#### Coverage by city')
        st.dataframe(frame,hide_index=True,use_container_width=True)
    rules=report.get('rules',{})
    if rules:
        st.markdown('#### Validation checks')
        items=list(rules.items()) if isinstance(rules,dict) else []
        if items:
            frame=pd.DataFrame([{'check':k,**(v if isinstance(v,dict) else {'result':v})} for k,v in items])
            st.dataframe(frame,hide_index=True,use_container_width=True)
    if not visible and not cities and not rules: st.info('Quality artifact contains no recognized summary, city coverage, or validation checks.')

def _monitoring_cards(report: dict):
    cols=st.columns(4)
    cols[0].metric('Monitoring status',str(report.get('status','unknown')).replace('_',' ').title())
    cols[1].metric('Matched samples',str(report.get('sample_count',0)))
    cols[2].metric('Model version',str(report.get('model_version') or 'not recorded'))
    cols[3].metric('Report generated',str(report.get('generated_at','not recorded')))
    for section in ('freshness','drift','performance'):
        value=report.get(section)
        if isinstance(value,dict) and value:
            st.markdown(f'#### {section.title()}')
            st.dataframe(pd.DataFrame([{'measure':k,'value':v} for k,v in value.items()]),hide_index=True,use_container_width=True)

def _metadata_cards(metadata: dict):
    cols=st.columns(4)
    cols[0].metric('Release',str(metadata.get('version','not recorded')))
    cols[1].metric('Supported cities',str(len(metadata.get('cities',[]))))
    dataset=metadata.get('dataset',{})
    cols[2].metric('Training dataset',str(dataset.get('name',dataset.get('source','not recorded')) if isinstance(dataset,dict) else dataset))
    cutoffs=metadata.get('cutoffs',{})
    cols[3].metric('Test cutoff',str(cutoffs.get('test_end',cutoffs.get('test','not recorded')) if isinstance(cutoffs,dict) else cutoffs))
    if metadata.get('cities'): st.caption('Custom model coverage: '+', '.join(map(str,metadata['cities'])))
    if metadata.get('features'): st.caption(f"{len(metadata['features'])} saved model features · temporal cutoffs recorded above")

def _plot(fig, height=310):
    fig.update_layout(height=height, margin=dict(l=4,r=8,t=15,b=5),paper_bgcolor='#111b24',plot_bgcolor='#111b24',font=dict(family='DM Sans',color='#91a3ad'),legend=dict(orientation='h',y=1.1),hoverlabel=dict(bgcolor='#14212b'))
    fig.update_xaxes(showgrid=False,zeroline=False,color='#91a3ad')
    fig.update_yaxes(gridcolor='#24323d',zeroline=False,color='#91a3ad')
    return fig

st.markdown('<div class="brand"><div class="brandmark"></div><div><span class="brandname">AIRCAST</span><span class="brandplace">INDIA · AIR QUALITY INTELLIGENCE</span></div></div>', unsafe_allow_html=True)

try:
    providers = _provider_module()
    provider_error = None
    cities, provider_error = _safe_call(providers, 'curated_cities')
    cities = cities or []
except Exception as exc:
    providers, cities, provider_error = None, [], f"{type(exc).__name__}: {exc}"

if not cities:
    cities = []

if providers:
    refresh_col, count_col = st.columns([1, 5])
    if refresh_col.button('Refresh provider data', help='Clear cached provider responses and reload locations and readings'):
        clear = getattr(providers, 'clear_cache', None)
        if callable(clear): clear()
        _provider_module.clear()
        st.cache_data.clear()
        st.rerun()
    count_col.caption(f'{len(cities)} locations in provider catalogue · provider readings refresh automatically when cache expires')

page = st.radio('Sections', ['Explore','Forecast','Compare Cities','Data Quality','Model Intelligence'], horizontal=True, label_visibility='collapsed')

# Search results retain full location objects; state is always included in labels.
st.markdown('<div class="eyebrow">Nationwide air quality · India</div>', unsafe_allow_html=True)
search_results = st.session_state.get('search_results', [])
city_pool = list(cities)
for item in search_results:
    if _city_id(item) not in {_city_id(c) for c in city_pool}: city_pool.insert(0,item)
if city_pool:
    labels = [_city_label(c) for c in city_pool]
    preferred = next((i for i, c in enumerate(city_pool) if _city_id(c) == 'chennai'), 0)
    if 'pending_city_label' in st.session_state:
        st.session_state['selected_city_label'] = st.session_state.pop('pending_city_label')
    if 'selected_city_label' not in st.session_state: st.session_state['selected_city_label'] = labels[preferred]
    selected_label = st.selectbox('Selected location', labels, key='selected_city_label', label_visibility='collapsed', accept_new_options=True, placeholder='Search a city or state')
    if selected_label not in labels:
        results, err = _safe_call(providers, 'search_cities', selected_label.strip())
        if err or not results:
            st.warning(f"No Indian locations found for {selected_label}. {err or ''}")
            selected = None
        else:
            st.session_state['search_results'] = results
            st.session_state['pending_city_label'] = _city_label(results[0])
            st.rerun()
    else:
        selected = city_pool[labels.index(selected_label)]
else:
    selected = None
    st.info('City discovery is unavailable because the provider has not returned locations. Check the configured data source and retry.')

snapshot = None
hourly = None
if selected and providers:
    snapshot, snap_err = _safe_call(providers, 'get_snapshot', selected)
    hourly, hour_err = _safe_call(providers, 'get_hourly', selected, past_days=7, forecast_days=3)
else: snap_err, hour_err = provider_error, provider_error

if page == 'Explore':
    if selected:
        name = _city_label(selected)
        aqi = getattr(snapshot,'aqi',None) if snapshot is not None else None
        aqi_text = f"{float(aqi):.0f}" if aqi is not None else '—'
        aqistd = getattr(snapshot,'aqi_standard',None) if snapshot is not None else None
        severity = 'Unavailable' if aqi is None else ('Good' if aqi<=50 else 'Moderate' if aqi<=100 else 'Unhealthy for sensitive groups' if aqi<=150 else 'Unhealthy' if aqi<=200 else 'Very unhealthy' if aqi<=300 else 'Hazardous')
        st.markdown(f'<div class="hero"><div><div class="eyebrow">Current conditions · provider data</div><h1>{name}</h1><p>Pollutant concentrations · source timestamp and freshness shown below</p></div><div class="hero-right"><div class="hero-num">{aqi_text}</div><div class="hero-sub">US AQI · {severity}</div></div></div>', unsafe_allow_html=True)
        if snapshot is None:
            st.warning(f"Current provider observations unavailable. {snap_err or 'No snapshot returned.'}")
        else:
            pollutant_order=['pm2_5','pm10','nitrogen_dioxide','sulphur_dioxide','carbon_monoxide','ozone']
            pollutants=getattr(snapshot,'pollutants',{}) or {}
            cols=st.columns(6)
            for col,key in zip(cols,pollutant_order):
                reading=pollutants.get(key) if isinstance(pollutants,dict) else None
                val=getattr(reading,'value',None) if reading is not None else None
                unit=getattr(reading,'unit','µg/m³') if reading is not None else 'µg/m³'
                text=f'{float(val):.1f}' if val is not None else '—'
                label='PM₂.₅' if key=='pm2_5' else key.replace('_',' ')
                col.markdown(f'<div class="reading"><div class="reading-name">{label}</div><div class="reading-value">{text} <span class="reading-unit">{unit}</span></div></div>',unsafe_allow_html=True)
            st.caption(f"{getattr(snapshot,'provider','Provider')} · {getattr(snapshot,'data_type','Provider forecast')} · source estimate {getattr(snapshot,'timestamp','unknown')} · retrieved {getattr(snapshot,'retrieved_at','unknown')} · freshness: {getattr(snapshot,'freshness','unknown')} · station: {getattr(snapshot,'station_id',None) or 'not identified'}")
        _section('Recent air quality', 'Actual provider hourly history · future forecast is shown separately')
        if hourly is not None and isinstance(hourly,pd.DataFrame) and not hourly.empty and 'time' in hourly.columns:
            data=hourly.copy(); data['time']=pd.to_datetime(data['time'],errors='coerce')
            data=data.dropna(subset=['time']); now_ist=pd.Timestamp.now(tz='Asia/Kolkata'); data['time_ist']=data['time'].dt.tz_localize('Asia/Kolkata') if data['time'].dt.tz is None else data['time'].dt.tz_convert('Asia/Kolkata')
            history=data[data.time_ist<=now_ist]
            value_cols=[c for c in ['pm25','pm10'] if c in data.columns]
            if value_cols:
                hist_cols=st.columns(2)
                with hist_cols[0]:
                    st.caption('OBSERVED / HISTORICAL')
                    fig=px.line(history,x='time_ist',y=value_cols,color_discrete_map={'pm25':'#58c7bd','pm10':'#83d9e8'},labels={'value':'Concentration (µg/m³)','time':'India Standard Time'})
                    st.plotly_chart(_plot(fig,300),use_container_width=True) if not history.empty else st.info('No historical hourly observations returned.')
                with hist_cols[1]:
                    st.caption('PROVIDER FORECAST')
                    future=data[data.time_ist>now_ist]
                    fig=px.line(future,x='time_ist',y=value_cols,color_discrete_map={'pm25':'#58c7bd','pm10':'#83d9e8'},labels={'value':'Concentration (µg/m³)','time':'India Standard Time'})
                    st.plotly_chart(_plot(fig,300),use_container_width=True) if not future.empty else st.info('No future forecast hours returned.')
                if 'us_aqi' in data: st.caption('AQI values, when available, follow the US AQI standard; this is not India’s NAQI.')
            else: st.info('The hourly provider response contains no PM₂.₅ or PM₁₀ series.')
        else: st.info(f"Hourly history and forecast unavailable. {hour_err or 'No provider data returned.'}")
        _section('Explore nearby coverage', 'Select an observed city point to inspect its provider snapshot.')
        if len(city_pool)>1:
            map_data=pd.DataFrame([_display_city(c) for c in city_pool]); map_data=map_data.dropna(subset=['latitude','longitude'])
            if not map_data.empty:
                map_data['label']=map_data.apply(lambda r:f"{r['name']}, {r['state']}",axis=1)
                map_data['selected']=map_data['id']==_city_id(selected)
                map_data['marker_size']=map_data['selected'].map({True:14,False:8})
                fig=px.scatter_map(map_data,lat='latitude',lon='longitude',hover_name='label',color='selected',color_discrete_map={True:'#83d9e8',False:'#58c7bd'},size='marker_size',size_max=15,zoom=5.5,center={'lat':float(_display_city(selected)['latitude']),'lon':float(_display_city(selected)['longitude'])},height=390)
                fig.update_layout(map_style='carto-darkmatter')
                ev=st.plotly_chart(_plot(fig,390),use_container_width=True,on_select='rerun',selection_mode='points',key='city_map')
                try:
                    point=ev.selection.points[0]
                    point_idx=point.get('point_index',point.get('pointNumber'))
                    map_label=map_data.iloc[int(point_idx)]['label']
                    if map_label in labels and map_label!=st.session_state.selected_city_label:
                        st.session_state.selected_city_label=map_label; st.rerun()
                except (AttributeError,IndexError,KeyError,TypeError,ValueError): pass
                st.caption('Map points represent curated provider locations, not ground stations. Select a point to update the city.')
        else: st.info('Map needs more than one available location.')
    else:
        st.markdown('<div class="hero"><div><div class="eyebrow">Nationwide air quality · India</div><h1>Air, made legible.</h1><p>Search Indian cities and inspect provider observations, hourly history, and model forecasts.</p></div></div>',unsafe_allow_html=True)
        st.info('City data is not currently reachable. The interface will not substitute example or fabricated readings.')
        if cities: st.write('Available locations', ', '.join(_city_label(c) for c in cities))
elif page == 'Forecast':
    _section('Forecast outlook','Provider atmospheric forecast and AirCast custom model are separate forecast products.')
    if not selected: st.info('Select a city above to view forecast coverage.')
    else:
        left,right=st.columns(2)
        with left:
            st.markdown('#### Provider forecast')
            if hourly is None or not isinstance(hourly,pd.DataFrame) or hourly.empty: st.info(f"Provider forecast unavailable. {hour_err or ''}")
            else:
                df=hourly.copy(); df['time']=pd.to_datetime(df['time'],errors='coerce'); df=df.dropna(subset=['time'])
                df['time_ist']=df['time'].dt.tz_localize('Asia/Kolkata') if df['time'].dt.tz is None else df['time'].dt.tz_convert('Asia/Kolkata')
                df=df[df.time_ist>=pd.Timestamp.now(tz='Asia/Kolkata')]
                cols=[c for c in ['pm25','pm10'] if c in df]
                if cols: st.plotly_chart(_plot(px.line(df,x='time_ist',y=cols,color_discrete_sequence=['#58c7bd','#83d9e8'],labels={'time_ist':'India Standard Time','value':'Concentration (µg/m³)'}),290),use_container_width=True)
                else: st.info('No particulate forecast series in provider response.')
                st.caption('Provider forecast is atmospheric model output, not a local sensor measurement.')
        with right:
            st.markdown('#### AirCast custom forecast')
            try:
                from aircast.probabilistic import predict_city, supported_cities
                supported=supported_cities()
                if _city_id(selected) not in supported: st.info('Custom model coverage does not include this city. Provider coverage may still be available.')
                else:
                    result,err=_safe_call(type('M',(),{'run':staticmethod(predict_city)}),'run',_city_id(selected))
                    if err: st.warning(f"Custom forecast unavailable: {err}")
                    elif result:
                        payload=result.model_dump() if hasattr(result,'model_dump') else dict(result)
                        st.metric('Predicted daily PM₂.₅',f"{payload.get('predicted_pm25',0):.1f} µg/m³")
                        st.caption(f"{payload.get('forecast_date','date unavailable')} · {payload.get('model_name','model unavailable')} {payload.get('model_version','')} · data cutoff {payload.get('data_cutoff','not recorded')}")
                        if payload.get('p10') is not None and payload.get('p90') is not None: st.caption(f"Calibrated interval: P10 {payload['p10']:.1f} – P90 {payload['p90']:.1f} µg/m³")
                        st.warning('Model-derived estimate, not a ground-station measurement. Check cutoff before interpreting freshness.')
            except (ImportError,AttributeError) as exc: st.info(f"Custom model artifacts or inference module are unavailable: {exc}")
elif page == 'Compare Cities':
    _section('Compare cities','Compare current provider pollutants; missing measurements remain missing.')
    options=[_city_label(c) for c in city_pool]
    chosen=st.multiselect('Cities to compare',options,default=options[:min(4,len(options))])
    if chosen and providers:
        by_label={_city_label(c):c for c in city_pool}; rows=[]
        for label in chosen:
            snap,err=_safe_call(providers,'get_snapshot',by_label[label])
            row={'city':label,'US AQI':getattr(snap,'aqi',None) if snap else None,'provider':getattr(snap,'provider','unavailable') if snap else 'unavailable','freshness':getattr(snap,'freshness','unknown') if snap else 'unavailable'}
            poll=getattr(snap,'pollutants',{}) if snap else {}
            for key in ['pm25','pm10','nitrogen_dioxide','sulphur_dioxide','carbon_monoxide','ozone']:
                reading=poll.get(key) if isinstance(poll,dict) else None; row[key.upper()]=getattr(reading,'value',None) if reading else None
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('AQI explicitly uses US AQI where supplied. Concentration units are as provided by each provider; no PM₂.₅-to-AQI conversion is performed.')
    else: st.info('Select one or more cities after provider locations load.')
elif page == 'Data Quality':
    _section('Data quality','Actual ingestion and quality reports only; unavailable monitoring is not interpreted as healthy.')
    report=_read_json(_artifact('artifacts/platform/quality.json'))
    provenance=_read_json(_artifact('artifacts/platform/provenance.json'))
    if not report: st.info('Quality report is not available. Live feed health and validation rules cannot currently be confirmed.')
    else:
        _quality_cards(report)
    with st.expander('Source provenance'):
        if provenance: st.json(provenance,expanded=True)
        else: st.info('Provenance artifact unavailable.')
    frame=_read_csv(_artifact('data/multicity_daily.csv'))
    parquet=_artifact('data/multicity_daily.parquet')
    if parquet.exists():
        try: frame=pd.read_parquet(parquet)
        except Exception as exc: st.warning(f'Could not read multicity parquet: {exc}')
    if frame is not None and not frame.empty:
        _section('Available historical records','Observed records and source-level completeness from the multicity dataset.')
        st.dataframe(frame.groupby('city_id').agg(records=('date','count'),first_date=('date','min'),last_date=('date','max'),median_hours=('hours','median')).reset_index(),hide_index=True,use_container_width=True)
    else: st.info('Multicity historical dataset is unavailable; no sample data is substituted.')
    monitoring=_read_json(_artifact('artifacts/platform/monitoring.json'))
    _section('Operational monitoring','Monitoring requires sufficient matured forecasts matched to subsequent real observations.')
    if monitoring: _monitoring_cards(monitoring)
    else: st.info('Insufficient monitoring evidence: no monitoring report is available.')
elif page == 'Model Intelligence':
    _section('Model intelligence','Quality, calibrated uncertainty, evaluation and provenance from actual saved artifacts.')
    manifest=_read_json(_artifact('artifacts/probabilistic/current.json'))
    if not manifest: st.info('Nationwide custom model release artifacts are not available. No model results are inferred or fabricated.')
    else:
        version_dir=manifest.get('directory') or manifest.get('path') or manifest.get('version_dir') or manifest.get('artifact_dir')
        base=(ROOT/'artifacts/probabilistic'/str(version_dir)).resolve() if version_dir else ROOT/'artifacts/probabilistic'
        if ROOT/'artifacts/probabilistic' not in base.parents and base != ROOT/'artifacts/probabilistic': base=ROOT/'artifacts/probabilistic'
        metadata=_read_json(base/'metadata.json')
        if metadata: _metadata_cards(metadata)
        for filename,title in [('comparison.csv','Model comparison'),('coverage.csv','Interval coverage'),('feature_importance.csv','Feature importance'),('backtest.csv','Chronological backtest')]:
            table=_read_csv(base/filename)
            with st.expander(title,expanded=filename=='comparison.csv'):
                if table is None or table.empty: st.info(f'{filename} is not available or contains no records.')
                elif filename=='comparison.csv':
                    city_col=next((c for c in ('city_id','city','location') if c in table.columns),None)
                    if city_col:
                        cities_in=sorted(table[city_col].dropna().astype(str).unique())
                        selected_cities=st.multiselect('Cities',cities_in,default=cities_in,key='comparison_city_filter')
                        table=table[table[city_col].astype(str).isin(selected_cities)]
                    st.dataframe(table,hide_index=True,use_container_width=True)
                    metric_col=next((c for c in ('mae','rmse','pinball_loss') if c in table.columns),None)
                    if metric_col and 'model' in table:
                        fig=px.bar(table,x='model',y=metric_col,color='model',color_discrete_sequence=['#58c7bd','#83d9e8','#91a3ad'])
                        st.plotly_chart(_plot(fig,250),use_container_width=True)
                else: st.dataframe(table,hide_index=True,use_container_width=True)
        predictions=_read_csv(base/'predictions.csv')
        if predictions is not None and not predictions.empty:
            st.markdown('#### Actual vs predicted')
            date_col='date' if 'date' in predictions else 'forecast_date' if 'forecast_date' in predictions else None
            city_col=next((c for c in ('city_id','city','location') if c in predictions.columns),None)
            if city_col:
                city_values=sorted(predictions[city_col].dropna().astype(str).unique())
                chosen=st.multiselect('Prediction cities',city_values,default=city_values,key='prediction_city_filter')
                predictions=predictions[predictions[city_col].astype(str).isin(chosen)]
            ycols=[x for x in ['actual','predicted_pm25','predicted'] if x in predictions]
            if date_col and ycols:
                predictions[date_col]=pd.to_datetime(predictions[date_col],errors='coerce')
                fig=go.Figure()
                if {'p10','p90'}.issubset(predictions.columns):
                    band=predictions.sort_values(date_col)
                    fig.add_trace(go.Scatter(x=band[date_col],y=band.p90,mode='lines',line=dict(width=0),showlegend=False,name='P90'))
                    fig.add_trace(go.Scatter(x=band[date_col],y=band.p10,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(88,199,189,.16)',name='P10–P90 interval'))
                for col,color in zip(ycols,['#58c7bd','#83d9e8','#91a3ad']): fig.add_trace(go.Scatter(x=predictions[date_col],y=predictions[col],mode='lines+markers',name=col.replace('_',' ').title(),line=dict(color=color,width=2)))
                st.plotly_chart(_plot(fig,300),use_container_width=True)
            else: st.info('Prediction artifact schema does not include plottable date and value columns.')

st.markdown('<div style="border-top:1px solid #24323d;margin-top:28px;padding-top:13px;color:#71838d;font-size:11px">AirCast · India · Provider coverage and custom model coverage are distinct. All concentrations are not AQI; AQI is identified as US AQI where provided. Model outputs are estimates, not ground measurements.</div>',unsafe_allow_html=True)
