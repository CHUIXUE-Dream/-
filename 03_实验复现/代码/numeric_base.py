"""v4: same-information strong baselines, local anchoring, dev-only routing and pooled calibration."""
from pathlib import Path
import sys,os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; V3=PROJECT/'v3'; sys.path.insert(0,str(PROJECT/'.deps'))
import numpy as np,pandas as pd
import json,hashlib,math,datetime,time,itertools
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
SCENARIOS=['mask1','mask2','mask4','block24','block72','donor_missing','bias1','bias2','relation_shift']; METHODS=['P_persistence','P_local_interp','P_seasonal','P_cross_absolute','P_delta_ridge','M_router','A_always_delta','A_no_local','A_no_cross']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def rmse(x): return float(np.sqrt(np.mean(np.square(x))))
def prep():
 y=pd.read_csv(V3/'results/hourly.csv.gz',index_col=0,parse_dates=True); ids=pd.read_csv(V3/'results/hourly_lineage.csv.gz',usecols=['hour','station','record_id'],dtype={'station':str}); ids['hour']=pd.to_datetime(ids.hour); ids=ids.pivot(index='hour',columns='station',values='record_id').reindex(y.index); return y,ids
def build(y,ids,year):
 cases=[]; gold=[]; times=y.index; arrays={s:y[s].to_numpy() for s in y}; timestamp_index={t:i for i,t in enumerate(times)}
 for s in y:
  dates=[d for d in pd.date_range(f'{year}-01-01',f'{year}-12-31',tz='UTC') if np.isfinite(arrays[s][[timestamp_index[d+pd.Timedelta(hours=h)] for h in [0,6,12,18]]]).all()]
  if year==2025:
   dates=[d for month in range(1,13) for d in sorted([a for a in dates if a.month==month],key=lambda d:hashlib.sha256(f'20261113|{s}|{d}'.encode()).hexdigest())[:12]]
  for day in sorted(dates):
   tt=[day+pd.Timedelta(hours=h) for h in [0,6,12,18]]; ix=[timestamp_index[t] for t in tt]; reference=arrays[s][ix]; order=sorted(range(4),key=lambda i:hashlib.sha256(f'{s}|{day}|{i}|20261113'.encode()).hexdigest())
   for scenario in SCENARIOS:
    if year in [2022,2023,2024] and scenario in ['bias1','bias2','relation_shift','block24']: continue
    missing=int(scenario[-1]) if scenario.startswith('mask') else 4; hidden=order[:missing]; boundary=timestamp_index[day]-(48 if scenario=='block72' else 0); lo=max(0,boundary-72); idx=np.flatnonzero(np.isfinite(arrays[s][lo:boundary])); anchor_idx=lo+int(idx[-1]) if len(idx) else None; anchor=float(arrays[s][anchor_idx]) if anchor_idx is not None else None
    donors={}; anchor_donors={}; source={}
    for j in y:
     if j==s: continue
     donors[j]=[float(v) if np.isfinite(v) else None for v in arrays[j][ix]]
     ai=None
     if anchor_idx is not None:
      dl=max(0,anchor_idx-6); di=np.flatnonzero(np.isfinite(arrays[j][dl:anchor_idx+1])); ai=dl+int(di[-1]) if len(di) else None
     anchor_donors[j]=float(arrays[j][ai]) if ai is not None else None; source[j]=dict(current=[str(ids.loc[t,j]) if pd.notna(ids.loc[t,j]) else None for t in tt],anchor=str(ids.iloc[ai][j]) if ai is not None else None)
     if scenario=='donor_missing' and j==sorted([a for a in y if a!=s])[0]: donors[j]=[None]*4; anchor_donors[j]=None
     if scenario in ['bias1','bias2']: donors[j]=[None if v is None else v+int(scenario[-1]) for v in donors[j]]
     if scenario=='relation_shift': donors[j]=[None if v is None else 50-v for v in donors[j]]
    cid=hashlib.sha256(f'v4|{s}|{day}|{scenario}'.encode()).hexdigest()[:24]; pub=dict(case_id=cid,station=s,date=str(day.date()),hours=[str(t) for t in tt],target_visible=[None if i in hidden else float(reference[i]) for i in range(4)],last_visible_value=anchor,last_visible_time=str(times[anchor_idx]) if anchor_idx is not None else None,donors=donors,donor_anchor=anchor_donors,donor_sources=source,target_anchor_source=str(ids.iloc[anchor_idx][s]) if anchor_idx is not None else None,target_visible_sources=[None if i in hidden else str(ids.loc[tt[i],s]) for i in range(4)],query_cutoff=str(day+pd.Timedelta(hours=19)))
    cases.append(pub); gold.append(dict(case_id=cid,station=s,date=str(day.date()),scenario=scenario,mask_count=missing,reference_mean=float(reference.mean()),reference_max=float(reference.max()),reference_values=reference.tolist(),reference_sources=[str(ids.loc[t,s]) for t in tt],data_kind='synthetic_shift' if scenario in ['bias1','bias2','relation_shift'] else 'real_masked'))
 return cases,pd.DataFrame(gold)
def state(c,models):
 s=c['station']; visible=c['target_visible']; missing=[i for i,v in enumerate(visible) if v is None]; m=len(missing); day=pd.Timestamp(c['date'],tz='UTC'); anchor=c['last_visible_value']; age=(day-pd.Timestamp(c['last_visible_time'])).total_seconds()/3600 if anchor is not None else 999; mode='long_gap' if age>=48 else 'short_gap'; alltimes=np.array([0.,6.,12.,18.]); obsx=[float(i*6) for i,v in enumerate(visible) if v is not None]; obsy=[v for v in visible if v is not None]
 if anchor is not None: obsx=[-age]+obsx; obsy=[anchor]+obsy
 local=np.interp(alltimes,obsx,obsy) if obsx else None
 persistence=np.array([v if v is not None else anchor for v in visible],dtype=float) if anchor is not None else None
 tt=pd.DatetimeIndex(c['hours']); u=(tt-pd.Timestamp('2022-01-01',tz='UTC')).total_seconds().to_numpy()/86400/365.2425*2*np.pi; base=models[s]; season=np.column_stack([np.ones(4)]+[f(k*u) for k in range(1,base['h']+1) for f in [np.sin,np.cos]])@base['seasonal_b']; season=np.array([v if v is not None else season[i] for i,v in enumerate(visible)])
 if local is None: local=season.copy()
 absolute=[]; delta=[]; donors=[]; reads=0
 for i in missing:
  ap=[]; weights=[]; changes=[]
  for j,vs in c['donors'].items():
   if vs[i] is None: continue
   e=base['edges'].get(j)
   if e and e['eligible']:
    bm=models[j]; sj=np.array([1]+[f(k*u[i]) for k in range(1,bm['h']+1) for f in [np.sin,np.cos]])@bm['seasonal_b']; ap.append(float(season[i]+e['b'][0]+e['b'][1]*(vs[i]-sj))); weights.append(max(e['rho'],0)**2/(e['dev_rmse']**2+1e-6))
   if c['donor_anchor'][j] is not None: changes.append((j,vs[i]-c['donor_anchor'][j])); donors.append(j)
   reads+=1
  absolute.append(float(np.average(ap,weights=weights)) if ap else None); delta.append(dict(changes))
 cross=np.array([v if v is not None else np.nan for v in visible],dtype=float)
 for i,p in zip(missing,absolute): cross[i]=p if p is not None else np.nan
 usable=sorted(set.intersection(*[set(d) for d in delta])) if delta else []; change=[float(np.mean([d[j] for d in delta])) for j in usable]
 # Numeric model features contain only visible local/neighbor changes, never hidden target reference.
 x=np.array([1.,min(age,120)/24,m/4,np.sin(u[0]),np.cos(u[0])]+[float(np.mean([d.get(j,0.) for d in delta])) for j in sorted(c['donors'])]+[float(j in usable) for j in sorted(c['donors'])])
 return dict(local=local,persistence=persistence,season=season,cross=cross,x=x,mask_count=m,mode=mode,donor_count=len(usable),anchor_age=age,edge_reads=reads,context=f"{s}|{m}|{mode}")
def fit(x,y,alpha):
 mean=x[:,1:].mean(axis=0); scale=x[:,1:].std(axis=0); scale[scale<1e-8]=1; a=np.column_stack([np.ones(len(x)),(x[:,1:]-mean)/scale]); penalty=np.eye(a.shape[1])*alpha; penalty[0,0]=0; b=np.linalg.solve(a.T@a+penalty,a.T@y); return dict(mean=mean.tolist(),scale=scale.tolist(),b=b.tolist(),alpha=alpha)
def pred(model,x): return float(np.r_[1,(x[1:]-model['mean'])/model['scale']]@model['b'])
def candidates(st,stat,reg):
 agg=np.mean if stat=='mean' else np.max
 local=float(agg(st['local'])); persistence=float(agg(st['persistence'])) if st['persistence'] is not None else None; seasonal=float(agg(st['season'])); cross=float(agg(st['cross'])) if np.isfinite(st['cross']).all() else None; delta=local+pred(reg,st['x']) if reg and st['donor_count']>0 else None
 return dict(P_local_interp=local,P_persistence=persistence,P_seasonal=seasonal,P_cross_absolute=cross,P_delta_ridge=delta)
def run():
 models=json.loads((V3/'results/models.json').read_text(encoding='utf-8')); y,ids=prep(); datasets={}; states={}
 for year in [2022,2023,2024,2025]:
  cases,gold=build(y,ids,year); datasets[year]=(cases,gold); states[year]={c['case_id']:state(c,models) for c in cases}; print('prepared',year,len(cases),flush=True)
 regressions={}; dev_predictions=[]
 for s in y:
  for stat in ['mean','max']:
   def design(year):
    cases,g=datasets[year]; g=g[g.station==s]; records=[(r,states[year][r.case_id]) for r in g.itertuples() if states[year][r.case_id]['donor_count']>0]; xx=np.array([st['x'] for r,st in records]); yy=np.array([getattr(r,'reference_'+stat)-(np.mean(st['local']) if stat=='mean' else np.max(st['local'])) for r,st in records]); return xx,yy
   x,r=design(2022); dx,dr=design(2023); options=[]
   for alpha in [1,10,100,1000]:
    reg=fit(x,r,alpha); error=np.array([pred(reg,v) for v in dx])-dr; options.append((rmse(error),reg))
   _,reg=min(options,key=lambda a:a[0]); regressions[s+'|'+stat]=reg
 for c in datasets[2023][0]:
  st=states[2023][c['case_id']]
  for stat in ['mean','max']: dev_predictions.append(dict(case_id=c['case_id'],station=c['station'],statistic=stat,context=st['context'],**candidates(st,stat,regressions[c['station']+'|'+stat])))
 dev=pd.DataFrame(dev_predictions).merge(datasets[2023][1],on=['case_id','station']); routes={}; selection=[]
 for (context,stat),g in dev.groupby(['context','statistic']):
  ref=g['reference_'+stat].to_numpy(); local=g.P_local_interp.to_numpy(); baseline=rmse(local-ref); options=[]
  for method in ['P_persistence','P_local_interp','P_seasonal','P_cross_absolute','P_delta_ridge']:
   for alpha in ([0.,.25,.5,.75,1.] if method in ['P_cross_absolute','P_delta_ridge'] else [1.]):
    candidate=g[method].to_numpy(); blend=np.where(np.isfinite(candidate),local*(1-alpha)+alpha*candidate,local); options.append((rmse(blend-ref),method,alpha))
  best=min(options,key=lambda a:(a[0],a[1],a[2])); chosen=best if len(g)>=30 and best[0]<baseline*.98 else (baseline,'P_local_interp',1.)
  routes[context+'|'+stat]=dict(method=chosen[1],alpha=chosen[2],n_dev=len(g),local_rmse=baseline,selected_rmse=chosen[0]); selection.append(dict(context=context,statistic=stat,**routes[context+'|'+stat]))
 dump(OUT/'regressions.json',regressions); dump(OUT/'routes.json',routes); pd.DataFrame(selection).to_csv(OUT/'dev_route_selection.csv',index=False)
 freeze=dict(version='v4',code_sha256=sha(Path(__file__)),parent_frozen_sha256=sha(V3/'results/frozen.json'),source_manifest_sha256=sha(V3/'data/manifest.json'),regressions_sha256=sha(OUT/'regressions.json'),routes_sha256=sha(OUT/'routes.json'),train=2022,dev=2023,cal=2024,evaluation=2025,evaluation_status='2025 previously inspected in v3; retrospective development-informed reevaluation, NOT new blind test',min_cal=100,selection_margin=.02,seed=20261113,interval_pooling='station x method x statistic x mask_count x anchor_age_bucket; mask positions pooled; donor shifts never used to calibrate',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 if (OUT/'frozen.json').exists():
  old=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
  for field in ['code_sha256','parent_frozen_sha256','source_manifest_sha256','regressions_sha256','routes_sha256']: assert old[field]==freeze[field],f'{field} changed; create a new version rather than silently overwrite'
 else: dump(OUT/'frozen.json',freeze)
 forecasts=[]; registry=[]
 for year,split in [(2024,'cal'),(2025,'reevaluation')]:
  cases,gold=datasets[year]; registry.append(gold.assign(split=split)); data=ROOT/'data/public'; data.mkdir(parents=True,exist_ok=True); (data/f'{split}.jsonl').write_text('\n'.join(json.dumps(c,ensure_ascii=False,allow_nan=False) for c in cases),encoding='utf-8'); data=ROOT/'data/gold'; data.mkdir(exist_ok=True); gold.to_csv(data/f'{split}.csv',index=False)
  for c in cases:
   st=states[year][c['case_id']]
   for stat in ['mean','max']:
    cand=candidates(st,stat,regressions[c['station']+'|'+stat]); route=routes.get(st['context']+'|'+stat,dict(method='P_local_interp',alpha=1.)); selected=cand.get(route['method']); alpha=route['alpha']; local=cand['P_local_interp']; blend=local if selected is None else local*(1-alpha)+selected*alpha
    outputs=dict(cand,M_router=blend,A_always_delta=cand['P_delta_ridge'] if cand['P_delta_ridge'] is not None else local,A_no_cross=cand[route['method']] if route['method'] in ['P_local_interp','P_persistence','P_seasonal'] and cand[route['method']] is not None else local,A_no_local=cand['P_cross_absolute'] if cand['P_cross_absolute'] is not None else cand['P_seasonal'])
    for method,value in outputs.items():
     if value is None: continue
     forecasts.append(dict(case_id=c['case_id'],station=c['station'],split=split,statistic=stat,method=method,candidate_value=value,mask_count=st['mask_count'],age_bucket=st['mode'],route=route['method'] if method=='M_router' else method,route_alpha=alpha if method=='M_router' else None,anchor_age_hours=st['anchor_age'],donor_count=st['donor_count'],edge_reads=st['edge_reads'] if method in ['P_delta_ridge','A_always_delta','A_no_local'] or (method=='M_router' and route['method'] in ['P_cross_absolute','P_delta_ridge'] and alpha>0) else 0))
 pd.DataFrame(forecasts).to_csv(OUT/'forecasts_without_gold.csv.gz',index=False); registry=pd.concat(registry); registry.to_csv(OUT/'case_registry.csv',index=False); z=pd.DataFrame(forecasts).merge(registry.drop(columns=['mask_count']),on=['case_id','station','split'],validate='many_to_one'); z['reference']=np.where(z.statistic=='mean',z.reference_mean,z.reference_max); z['error']=z.candidate_value-z.reference; z['q']=np.nan; z['n_cal']=0
 for key,ix in z.groupby(['station','method','statistic','mask_count','age_bucket']).groups.items():
  a=z.loc[ix]; cal=a[(a.split=='cal')&a.scenario.isin(['mask1','mask2','mask4','block72'])].drop_duplicates(['date','station','mask_count','age_bucket']); r=cal.error.abs().sort_values(); n=len(r); k=math.ceil((n+1)*.9); z.loc[ix,'n_cal']=n
  if n>=100 and k<=n: z.loc[ix,'q']=r.iloc[k-1]
 z['eligible']=z.q.notna(); z['kind']=np.where(z.eligible,'estimate','hypothesis'); z['numeric_value']=z.candidate_value.where(z.eligible); z['covered']=z.error.abs()<=z.q; z['width']=z.q*2; z.to_csv(OUT/'predictions.csv.gz',index=False)
 metrics=[]; test=z[z.split=='reevaluation']; universe=registry[registry.split=='reevaluation']
 for scenario in SCENARIOS:
  for stat in ['mean','max']:
   for method in METHODS:
    g=test[(test.scenario==scenario)&(test.statistic==stat)&(test.method==method)]; delivered=g[g.eligible]; n=len(universe[universe.scenario==scenario]); metrics.append(dict(scenario=scenario,statistic=stat,method=method,n_requests=n,n_candidate=len(g),n_delivered=len(delivered),delivery_rate=len(delivered)/n,RMSE=rmse(g.error) if len(g) else None,MAE=float(g.error.abs().mean()) if len(g) else None,PICP=float(delivered.covered.mean()) if len(delivered) else None,MPIW=float(delivered.width.mean()) if len(delivered) else None,n_interval=len(delivered),edge_reads=float(g.edge_reads.mean()) if len(g) else None,within_05_per_request=float((delivered.error.abs()<=.5).sum()/n),within_1_per_request=float((delivered.error.abs()<=1).sum()/n)))
 pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
 # Fair v3-v4 comparison by station/date/scenario/statistic, with identical old requests.
 old=pd.read_csv(V3/'results/predictions.csv.gz',dtype={'station':str}); old=old[(old.split=='test')&old.method.isin(['M_est','P1_persistence','P4_ridge'])]; combined=pd.concat([test[['station','date','scenario','statistic','method','error','eligible']].assign(version='v4'),old[['station','date','scenario','statistic','method','error','eligible']].assign(version='v3')]); combined.to_csv(OUT/'cross_version_errors.csv.gz',index=False)
 # Date-block paired uncertainty against strong local baseline, not just the old weak main.
 boot=[]; rng=np.random.default_rng(20261113)
 for scenario in ['mask1','mask2','mask4','block72','donor_missing']:
  g=test[(test.scenario==scenario)&(test.statistic=='mean')&test.method.isin(['P_local_interp','M_router','P_persistence','P_delta_ridge'])]; p=g.pivot(index=['station','date'],columns='method',values='error'); p=p.dropna(subset=['P_local_interp','M_router','P_persistence']); p.to_csv(OUT/f'paired_{scenario}.csv'); a=p.reset_index(); a['date']=pd.to_datetime(a.date); dates=pd.date_range('2025-01-01','2025-12-31'); cols=['P_local_interp','M_router','P_persistence']; ss=a.groupby('date')[cols].agg(lambda s:float((s*s).sum())).reindex(dates,fill_value=0).to_numpy(); nn=a.groupby('date').size().reindex(dates,fill_value=0).to_numpy()
  for block in [3,7,14]:
   draws=[]
   for _ in range(2000):
    starts=rng.integers(0,len(dates)-block+1,size=math.ceil(len(dates)/block)); ix=np.concatenate([np.arange(i,i+block) for i in starts])[:len(dates)]; n=nn[ix].sum()
    if n: draws.append(np.sqrt(ss[ix].sum(axis=0)/n))
   d=np.array(draws)
   for j,method in enumerate(cols[1:],1):
    diff=d[:,j]-d[:,0]; boot.append(dict(scenario=scenario,method=method,reference='P_local_interp',n=len(p),block_days=block,delta_RMSE=rmse(p[method])-rmse(p.P_local_interp),lower=float(np.quantile(diff,.025)),upper=float(np.quantile(diff,.975))))
 pd.DataFrame(boot).to_csv(OUT/'bootstrap.csv',index=False)
 # Preference sensitivity: precision is a user preference, not chosen from test labels.
 preference=[]
 for tau in [.25,.5,1.,2.,float('inf')]:
  for scenario in SCENARIOS:
   g=test[(test.method=='M_router')&(test.statistic=='mean')&(test.scenario==scenario)]; a=g[g.eligible&(g.q<=tau)]; preference.append(dict(scenario=scenario,max_halfwidth_degC=tau,n_requests=len(g),n_delivered=len(a),delivery_rate=len(a)/len(g) if len(g) else 0,MAE=float(a.error.abs().mean()) if len(a) else None,PICP=float(a.covered.mean()) if len(a) else None))
 pd.DataFrame(preference).to_csv(OUT/'precision_preferences.csv',index=False)
 checks=dict(public_hidden_values_absent=True,calibration_excludes_all_perturbations=True,mask_position_pooling_prespecified=True,min_cal_support=bool(test.loc[test.eligible,'n_cal'].ge(100).all()),code_matches_freeze=sha(Path(__file__))==freeze['code_sha256'],v3_untouched=sha(V3/'results/frozen.json')==freeze['parent_frozen_sha256'],new_blind_test=False)
 for c in datasets[2025][0]:
  assert all(v is None for v,r in zip(c['target_visible'],c['target_visible_sources']) if r is None)
  if c['last_visible_time']: assert pd.Timestamp(c['last_visible_time'])<pd.Timestamp(c['hours'][0])
 dump(OUT/'verification.json',checks); print(pd.DataFrame(metrics).query("scenario in ['mask1','mask2','mask4','block72','donor_missing'] and statistic=='mean' and method in ['P_local_interp','P_persistence','M_router']").to_string(index=False),flush=True)
if __name__=='__main__': run()
