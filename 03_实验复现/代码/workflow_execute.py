"""Execute the frozen E5 rule on cache misses; post-test implementation consistency check."""
from pathlib import Path
import sys,ast,json,hashlib,sqlite3,math
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
source=(P/'v3/cache_experiment.py').read_text(encoding='utf-8');node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Cache');exec(ast.get_source_segment(source,node),globals())
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
cal=json.loads((R/'results/calibration.json').read_text());freeze=json.loads((R/'results/frozen.json').read_text());cases=json.loads((R/'data/public.json').read_text());gold=pd.read_json(R/'data/gold.json',dtype={'station':str}).set_index('case_id');z=pd.read_csv(R/'results/point_predictions.csv.gz');base=z[(z.method=='v4')&(z.statistic=='mean')].set_index('case_id');events=pd.read_csv(R/'results/workflow_events.csv.gz');expected=events[(events.policy=='T_adapt4')&(events.tau==.25)].set_index('case_id');reads=[]
def execute(c,plan):
 known={};unseen=list(range(4));day=pd.Timestamp(c['date'],tz='UTC');anchor=c['last_visible_value'];at=(pd.Timestamp(c['last_visible_time'])-day).total_seconds()/3600 if anchor is not None else -24
 for b in range(plan['cap']+1):
  q=cal[f"{c['station']}|{plan['policy']}|{b}"]['q'];xs=([at] if anchor is not None else [])+[i*6 for i in known];vs=([anchor] if anchor is not None else [])+[known[i] for i in known];order=np.argsort(xs)
  pred=float(base.loc[c['case_id']].prediction) if b==0 else float(np.mean(np.interp([0,6,12,18],np.array(xs)[order],np.array(vs)[order])))
  delivered=b==4 or (q is not None and q<=plan['tau'])
  if delivered or b==plan['cap']:return dict(value=pred if delivered else None,kind='direct' if b==4 else 'estimate' if delivered else 'unresolved',budget=b,q=q)
  chosen=max(unseen,key=lambda i:(min(abs(i*6-t) for t in xs) if xs else 999,-i));known[chosen]=float(gold.loc[c['case_id']].reference_values[chosen]);unseen.remove(chosen);reads.append(dict(case_id=c['case_id'],step=b+1,hour=chosen*6,source=gold.loc[c['case_id']].reference_sources[chosen]))
cache=Cache('A_no_hotness');hits=plans=bad=0;ix=0
for c in cases:
 keydata=dict(case=c['case_id'],station=c['station'],hours=c['hours'],tau=.25,stat='mean',kind='estimate_allowed',calibration=freeze['calibration_sha256'],rule=freeze['code_sha256']);key=h(keydata);pk=h({k:v for k,v in keydata.items() if k not in ['case','hours']})
 for repeat in range(2):
  ans=cache.get('r'+key,ix);hits+=ans is not None
  if ans is None:
   plan=cache.get('p'+pk,ix);plans+=plan is not None
   if plan is None:plan=dict(policy='T_adapt4',tau=.25,cap=4);cache.put('p'+pk,'path',plan,1,ix)
   ans=execute(c,plan);cache.put('r'+key,'result',ans,1,ix)
  r=expected.loc[c['case_id']];bad+=not(ans['kind']==r.kind and ans['budget']==r.budget and math.isclose(ans['value'],r.prediction,abs_tol=1e-12));ix+=1
pd.DataFrame(reads).to_csv(R/'results/executed_acquisition_reads.csv',index=False)
out=dict(passed=bad==0,mismatches=bad,executions=len(cases),result_hits=hits,path_hits=plans,requests=ix,acquisition_reads=len(reads),scope='Post-test consistency replay of frozen decision rules with actual acquisition reads on cache misses; not a new holdout or online model latency benchmark.')
(R/'results/execution_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out));assert bad==0
