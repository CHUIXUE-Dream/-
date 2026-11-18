"""Publication reanalysis of frozen outputs; no refit or test-dependent point routing."""
from pathlib import Path
import sys,json,hashlib,math
P=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
O=P/'report_work/thesis05';O.mkdir(exist_ok=True);A=O/'analysis';A.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
freeze_paths=[P/v/'results/frozen.json' for v in ['v3','v4','v5','v6']];before={str(p.relative_to(P)):sha(p) for p in freeze_paths}
cohorts={'2021':P/'v6/results/predictions.csv.gz','2026':P/'v5/results/predictions.csv.gz'}
cal=json.loads((P/'v5/results/calibration.json').read_text());out=[];point=[];winners=[]
for cohort,path in cohorts.items():
 z=pd.read_csv(path,dtype={'station':str},low_memory=False)
 # Main point predictions remain the existing v4 route, selected in 2023.
 main=z[z.method=='v4'].copy()
 for policy in ['fixed','conditional']:
  f=main.copy();f['cohort']=cohort;f['interval_policy']=policy
  if policy=='conditional':
   for context,idx in f.groupby('context').groups.items():
    q=cal[context+'|v4']['q'];f.loc[idx,'lower']=f.loc[idx,'lower_raw']-q if q is not None else np.nan;f.loc[idx,'upper']=f.loc[idx,'upper_raw']+q if q is not None else np.nan
  f['covered']=(f.reference>=f.lower)&(f.reference<=f.upper);f['width']=f.upper-f.lower
  f['interval_score']=f.width+20*np.maximum(f.lower-f.reference,0)+20*np.maximum(f.reference-f.upper,0)
  out.append(f)
 # Preserve all existing candidates when identifying observed best values.
 f=z[z.method.isin(['local','persistence','v4','v5','v5_no_guard','v6'])].copy()
 for (scene,stat,method),g in f.groupby(['scenario','statistic','method']):
  point.append(dict(cohort=cohort,scenario=scene,statistic=stat,method=method,n=len(g),RMSE=float(np.sqrt(np.mean(g.error**2))),MAE=float(abs(g.error).mean())))
 df=pd.DataFrame([r for r in point if r['cohort']==cohort])
 for (scene,stat),g in df.groupby(['scenario','statistic']):
  for metric in ['RMSE','MAE']:
   value=float(g[metric].min());win=g.loc[np.isclose(g[metric],value,rtol=0,atol=1e-12),'method'].tolist();winners.append(dict(cohort=cohort,scenario=scene,statistic=stat,metric=metric,best_value=value,best_methods='|'.join(win),n=int(g.n.iloc[0])))
same=pd.concat(out,ignore_index=True);same.to_csv(A/'same_point_intervals.csv.gz',index=False)
metrics=same.groupby(['cohort','scenario','statistic','interval_policy']).agg(n=('case_id','size'),RMSE=('error',lambda a:float(np.sqrt(np.mean(a*a)))),MAE=('error',lambda a:float(abs(a).mean())),PICP=('covered','mean'),MPIW=('width','mean'),interval_score=('interval_score','mean')).reset_index();metrics.to_csv(A/'interval_metrics.csv',index=False)
pd.DataFrame(point).to_csv(A/'point_metrics.csv',index=False);pd.DataFrame(winners).to_csv(A/'observed_best.csv',index=False)
preferences=[]
for (cohort,scene,stat,policy),g in same.groupby(['cohort','scenario','statistic','interval_policy']):
 for tau in [.25,.5,1.,2.]:
  f=g[g.width/2<=tau];preferences.append(dict(cohort=cohort,scenario=scene,statistic=stat,interval_policy=policy,halfwidth=tau,n_requests=len(g),n_delivered=len(f),delivery_rate=len(f)/len(g),within05_per_request=float(f.error.abs().le(.5).sum()/len(g)),selected_PICP=float(f.covered.mean()) if len(f) else None))
pd.DataFrame(preferences).to_csv(A/'interval_preferences.csv',index=False)
# Main-vs-local date-block paired intervals on each cohort, not pooled across years.
rng=np.random.default_rng(20261113);boots=[]
for cohort,path in cohorts.items():
 z=pd.read_csv(path,dtype={'station':str},low_memory=False)
 for stat in ['mean','max']:
  g=z[(z.scenario=='mask4')&(z.statistic==stat)&z.method.isin(['local','v4','v5','v5_no_guard'])]
  pivot=g.pivot(index=['date','station'],columns='method',values='error').reset_index();pivot['date']=pd.to_datetime(pivot.date);dates=pd.date_range(pivot.date.min(),pivot.date.max());cols=['local','v4','v5','v5_no_guard'];ss=pivot.groupby('date')[cols].agg(lambda a:float((a*a).sum())).reindex(dates,fill_value=0).to_numpy();nn=pivot.groupby('date').size().reindex(dates,fill_value=0).to_numpy()
  for block in [3,7,14]:
   draw=[]
   for _ in range(2000):
    starts=rng.integers(0,len(dates)-block+1,size=math.ceil(len(dates)/block));ix=np.concatenate([np.arange(i,i+block) for i in starts])[:len(dates)];draw.append(np.sqrt(ss[ix].sum(0)/nn[ix].sum()))
   draw=np.array(draw)
   for a,b in [(1,0),(2,1),(3,1)]:
    diff=draw[:,a]-draw[:,b];boots.append(dict(cohort=cohort,statistic=stat,method=cols[a],reference=cols[b],n=len(pivot),block_days=block,delta_RMSE=float(np.sqrt(np.mean(pivot[cols[a]]**2))-np.sqrt(np.mean(pivot[cols[b]]**2))),lower=float(np.quantile(diff,.025)),upper=float(np.quantile(diff,.975))))
pd.DataFrame(boots).to_csv(A/'paired_intervals.csv',index=False)
checks={'frozen_versions_unchanged':all(sha(P/name)==digest for name,digest in before.items()),'same_points_between_interval_policies':bool(same.pivot(index=['cohort','case_id','statistic'],columns='interval_policy',values='prediction').diff(axis=1).iloc[:,1].abs().le(1e-12).all()),'all_interval_cells':len(metrics)==2*9*2*2,'same_denominators':bool(metrics.apply(lambda r:r.n==(922 if r.cohort=='2021' else 80),axis=1).all()),'no_refit':True}
dump(A/'analysis_audit.json',{'checks':checks,'passed':sum(checks.values()),'total':len(checks),'scope':'Post hoc publication reanalysis of frozen outputs, not a new held-out experiment. Fixed and conditional policies use identical v4 points.','sources':{str(p.relative_to(P)):sha(p) for p in list(cohorts.values())+[P/'v5/results/calibration.json']},'retained_point_method':'v4 development-selected main route; individual metric minima remain separately reported'});assert all(checks.values())
print(metrics.query("scenario=='mask4' and statistic=='mean'").to_string(index=False));print(pd.DataFrame(boots).query("block_days==7").to_string(index=False));print(pd.DataFrame(winners).query("scenario=='mask4'").to_string(index=False))
