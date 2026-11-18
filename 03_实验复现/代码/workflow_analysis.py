from pathlib import Path
import sys,json,math
P=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
O=P/'v7/results';z=pd.read_csv(O/'workflow_events.csv.gz',dtype={'station':str});rng=np.random.default_rng(20261114);rows=[]
for tau in [.1,.25,.5]:
 for method,reference in [('T_adapt2','H_adapt2'),('T_adapt2','T_fixed2'),('T_adapt4','all4')]:
  g=z[(z.tau==tau)&z.policy.isin([method,reference])];p=g.pivot(index=['date','station'],columns='policy',values=['qualified','budget']).reset_index();dates=pd.date_range(p.date.min(),p.date.max());p['date']=pd.to_datetime(p['date']);nn=p.groupby('date').size().reindex(dates,fill_value=0).to_numpy();v=[]
  for metric in ['qualified','budget']:v.append((p[(metric,method)].astype(float)-p[(metric,reference)].astype(float)).groupby(p.date).sum().reindex(dates,fill_value=0).to_numpy())
  a=np.array(v).T;draw=[]
  for _ in range(2000):
   starts=rng.integers(0,len(dates)-6,size=math.ceil(len(dates)/7));ix=np.concatenate([np.arange(s,s+7) for s in starts])[:len(dates)];draw.append(a[ix].sum(0)/nn[ix].sum())
  draw=np.array(draw)
  for j,m in enumerate(['qualified_rate','avg_records']):rows.append(dict(tau=tau,method=method,reference=reference,metric=m,difference=float(a[:,j].sum()/nn.sum()),lower=float(np.quantile(draw[:,j],.025)),upper=float(np.quantile(draw[:,j],.975)),n=len(p),block_days=7,repeats=2000))
pd.DataFrame(rows).to_csv(O/'paired_workflow.csv',index=False)
g=pd.read_csv(O/'workflow_metrics.csv');out=[]
for r in g.itertuples():
 for lam in [0,.01,.05,.1]:out.append(dict(policy=r.policy,tau=r.tau,lambda_per_record=lam,utility=r.qualified_rate-lam*r.avg_records))
pd.DataFrame(out).to_csv(O/'cost_sensitivity.csv',index=False)
point=pd.read_csv(O/'point_predictions.csv.gz');point.groupby(['statistic','method']).agg(n=('error','size'),RMSE=('error',lambda a:float(np.sqrt(np.mean(a*a)))),MAE=('error',lambda a:float(abs(a).mean()))).reset_index().to_csv(O/'point_metrics.csv',index=False)
print(pd.DataFrame(rows).query('tau==.25').to_string(index=False));print(pd.read_csv(O/'point_metrics.csv').to_string(index=False))
