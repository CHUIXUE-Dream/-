from pathlib import Path
import sys,os,json,hashlib,datetime,gzip,urllib.request,math
R=Path(__file__).parent;P=R.parent;sys.path[:0]=[str(P/'.deps'),str(P/'v5')]
import run_experiment as e
import pandas as pd,numpy as np
D=R/'data';O=R/'results';D.mkdir(exist_ok=True);O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def apply_policy(z):
    v=z[z.method=='v5'].copy();use=(v.statistic=='mean')&v.context.str.contains('short_gap')&(~v.fallback.astype(bool))
    v['interval_route']=np.where(use,'conditional','fixed');v['lower']=np.where(use,v.lower,v.fixed_lower);v['upper']=np.where(use,v.upper,v.fixed_upper);v['covered']=(v.reference>=v.lower)&(v.reference<=v.upper);v['width']=v.upper-v.lower;v['method']='v6'
    return pd.concat([z,v],ignore_index=True)
def metrics(z,path):
    rows=[]
    for key,g in z.groupby(['scenario','statistic','method']):
        valid=g[g.lower.notna()];score=valid.width+20*np.maximum(valid.lower-valid.reference,0)+20*np.maximum(valid.reference-valid.upper,0)
        rows.append(dict(scenario=key[0],statistic=key[1],method=key[2],n=len(g),RMSE=e.rmse(g.error),MAE=float(g.error.abs().mean()),PICP=float(valid.covered.mean()) if len(valid) else None,MPIW=float(valid.width.mean()) if len(valid) else None,interval_score=float(score.mean()) if len(valid) else None,n_interval=len(valid)))
    pd.DataFrame(rows).to_csv(path,index=False)
if not (O/'frozen.json').exists():
    # This occurs BEFORE retrieving the unseen 2021 bytes.
    dump(O/'frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=sha(__file__),protocol_sha256=sha(R/'PROTOCOL.md'),v5_freeze_sha256=sha(P/'v5/results/frozen.json'),evaluation_year=2021,interpretation='newly acquired held-out earlier year; backward temporal transfer, not prospective forecasting'))
freeze=json.loads((O/'frozen.json').read_text());assert sha(__file__)==freeze['code_sha256']
raw=D/'raw';raw.mkdir(exist_ok=True);manifest=[];lines=[]
for s in e.STATIONS:
    path=raw/f'{s}h2021.txt.gz';url=f'https://www.ndbc.noaa.gov/data/historical/stdmet/{s}h2021.txt.gz'
    if path.exists():
        b=path.read_bytes();prior=json.loads((D/'source_manifest.json').read_text()) if (D/'source_manifest.json').exists() else [];retrieved=next((a['retrieved_utc'] for a in prior if a['station']==s),None)
    else:
        b=urllib.request.urlopen(url,timeout=45).read();assert gzip.decompress(b).startswith(b'#');path.write_bytes(b);retrieved=datetime.datetime.now(datetime.timezone.utc).isoformat()
    h=hashlib.sha256(b).hexdigest();manifest.append(dict(station=s,year=2021,url=url,sha256=h,bytes=len(b),retrieved_utc=retrieved))
    text=gzip.decompress(b).decode('ascii');head=text.splitlines()[0].lstrip('#').split();wt=head.index('WTMP')
    for ln,line in enumerate(text.splitlines(),1):
        if line.startswith('#') or not line.strip():continue
        a=line.split()
        try:
            value=float(a[wt]);assert -5<=value<=45
            tt=pd.Timestamp(year=int(a[0]),month=int(a[1]),day=int(a[2]),hour=int(a[3]),minute=int(a[4]),tz='UTC');lines.append(dict(station=s,time=tt,hour=tt.floor('h'),value=value,record_id=f'v6:{s}:{ln}',line=ln,source_sha256=h))
        except (ValueError,AssertionError):continue
    dump(D/'source_manifest.json',manifest)
f=pd.DataFrame(lines).sort_values('time').drop_duplicates(['station','hour'],keep='last');f.to_csv(D/'hourly_lineage.csv.gz',index=False);ix=pd.date_range('2021-01-01','2022-01-01',freq='h',tz='UTC',inclusive='left');y=f.pivot(index='hour',columns='station',values='value').reindex(index=ix,columns=e.STATIONS);ids=f.pivot(index='hour',columns='station',values='record_id').reindex(index=ix,columns=e.STATIONS);y.to_csv(D/'hourly.csv.gz')
cases,gold=e.build_snapshot(y,ids);dump(D/'public.json',cases);gold.to_json(D/'gold.json',orient='records',force_ascii=False,indent=2)
bundle=e.load_bundle();records=[]
for c in cases:
    st=e.old.state(c,bundle['base']);x,od=e.features(c,st);records.append((c,st,x,od))
forecast=e.forecast(records,bundle);forecast.to_csv(O/'forecasts_without_gold.csv.gz',index=False);z=e.attach_gold(forecast,gold);cal=json.loads((P/'v5/results/calibration.json').read_text());z['n_cal']=0
for c in ['lower','upper','fixed_lower','fixed_upper']:z[c]=np.nan
for (context,method),i in z.groupby(['context','method']).groups.items():
    q=cal[context+'|'+method];z.loc[i,'n_cal']=q['n']
    if q['q'] is not None:
        z.loc[i,'lower']=z.loc[i,'lower_raw']-q['q'];z.loc[i,'upper']=z.loc[i,'upper_raw']+q['q'];z.loc[i,'fixed_lower']=z.loc[i,'prediction']-q['q_fixed'];z.loc[i,'fixed_upper']=z.loc[i,'prediction']+q['q_fixed']
oldp=pd.read_csv(P/'v4/results/predictions.csv.gz',dtype={'station':str});oldp=oldp[(oldp.split=='cal')&(oldp.method=='M_router')].drop_duplicates(['station','mask_count','age_bucket','statistic'])
for row in oldp.itertuples():
    context=f'{row.station}|{int(row.mask_count)}|{row.age_bucket}|{row.statistic}';i=z.index[(z.context==context)&(z.method=='v4')];z.loc[i,'lower']=z.loc[i,'prediction']-row.q;z.loc[i,'upper']=z.loc[i,'prediction']+row.q
z['covered']=(z.reference>=z.lower)&(z.reference<=z.upper);z['width']=z.upper-z.lower;z=apply_policy(z);z.to_csv(O/'predictions.csv.gz',index=False);metrics(z,O/'metrics.csv')
recent=apply_policy(pd.read_csv(P/'v5/results/predictions.csv.gz',dtype={'station':str}));recent.to_csv(O/'recent_retrospective_predictions.csv.gz',index=False);metrics(recent,O/'recent_retrospective_metrics.csv')
# Reuse identical acquisition protocol, write files to this new version only.
e.O=O;e.acquire(cases,gold);e.bootstrap(z)
claims=[];audits=[]
for row in z[z.method=='v6'].itertuples():
    cid=row.case_id+'|'+row.statistic;claims.append(dict(claim_id=cid,version='v6',kind='estimate' if pd.notna(row.lower) else 'hypothesis',value=row.prediction,lower=row.lower if pd.notna(row.lower) else None,upper=row.upper if pd.notna(row.upper) else None,interval_route=row.interval_route,namespace=freeze['code_sha256']));audits.append(dict(claim_id=cid,reference=row.reference,error=row.error,covered=bool(row.covered),operation='append verification; original preserved'))
for name,content in [('claims.jsonl',claims),('delayed_verification.jsonl',audits)]: (O/name).write_text('\n'.join(json.dumps(x,ensure_ascii=False,allow_nan=False) for x in content),encoding='utf-8')
checks={'rule_code_matches_freeze':sha(__file__)==freeze['code_sha256'],'new_data_downloaded_after_freeze':all(m['retrieved_utc']>freeze['created_utc'] for m in manifest),'all108_metric_cells':len(pd.read_csv(O/'metrics.csv'))==108,'claims_and_verification_one_to_one':len(claims)==len(audits)==len(gold)*2,'max_and_longgap_use_fixed':bool(z[(z.method=='v6')&((z.statistic=='max')|z.context.str.contains('long_gap'))].interval_route.eq('fixed').all()),'no_point_changes_v5_to_v6':bool(z[z.method.isin(['v5','v6'])].pivot(index=['case_id','statistic'],columns='method',values='prediction').diff(axis=1).iloc[:,1].abs().le(1e-12).all()),'full_acquisition_restores_exact':bool(pd.read_csv(O/'acquisition_events.csv').query('budget==4').error.abs().le(1e-12).all())}
dump(O/'audit.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks)));assert all(checks.values())
dump(O/'summary.json',dict(normal_requests=int((gold.scenario=='mask4').sum()),total_requests=len(gold),by_station=gold[gold.scenario=='mask4'].groupby('station').size().to_dict(),request_dates=[gold.date.min(),gold.date.max()],interpretation=freeze['interpretation']))
print(json.dumps(json.loads((O/'summary.json').read_text()),indent=2));print(pd.read_csv(O/'metrics.csv').query("scenario in ['mask4','block72','bias2'] and method in ['local','v4','v5','v6']").to_string(index=False))
