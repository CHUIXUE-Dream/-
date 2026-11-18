"""Deterministic evidence study; never represented as isolated ChatGPT efficacy."""
from pathlib import Path
import sys,json,copy,itertools,hashlib,unittest,io
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent/'.deps')); sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT/'compat'))
import numpy as np,pandas as pd
OUT=ROOT/'results'
def solve(scope,times,stat,es,raw_only=False,scope_off=False):
 if not times or len(set(times))!=len(times): return dict(kind='unresolved',value=None)
 target=set(times); observations={}; candidates=[]; groups=[]
 for original in es:
  e=copy.deepcopy(original)
  if scope_off: e['scope']=scope
  if e['scope']!=scope: continue
  ts=e['times']; unit=e.get('unit','degC'); typ=e['type']
  if unit not in ['degC','degF']: continue
  def cv(x): return (F(str(x))-32)*5/9 if unit=='degF' else F(str(x))
  if typ=='raw':
   if len(ts)!=len(e['values']): continue
   for t,v in zip(ts,e['values']):
    if t in target:
     observations.setdefault(t,[]).append((cv(v),e['id']))
  elif not raw_only and len(set(ts))==len(ts):
   if set(ts)==target and typ==stat: candidates.append((cv(e['value']),[e['id']],typ))
   if set(ts)==target and typ=='sum' and stat=='mean' and e.get('n')==len(target) and unit=='degC': candidates.append((F(str(e['value']))/e['n'],[e['id']],'sum/n'))
   if typ=='group_'+stat and e.get('n')==len(ts) and set(ts)<=target and ts: groups.append(e)
 if any(len({v for v,i in vals})>1 for vals in observations.values()): return dict(kind='conflict',value=None)
 if set(observations)==target:
  vals=[observations[t][0][0] for t in times]; ids=sorted({i for t in times for v,i in observations[t]}); candidates.append((sum(vals)/len(vals) if stat=='mean' else max(vals),ids,'raw'))
 for n in range(1,min(8,len(groups))+1):
  for subset in itertools.combinations(groups,n):
   seen=set(); valid=True
   for e in subset:
    if seen&set(e['times']): valid=False; break
    seen|=set(e['times'])
   if valid and seen==target:
    vals=[F(str(e['value'])) for e in subset]
    value=sum(v*e['n'] for v,e in zip(vals,subset))/len(target) if stat=='mean' else max(vals)
    candidates.append((value,[e['id'] for e in subset],'groups'))
 if not candidates: return dict(kind='unresolved',value=None)
 if len({v for v,_,_ in candidates})>1: return dict(kind='conflict',value=None)
 value,ids,path=candidates[0]; return dict(kind='direct',value=float(value),certificate=dict(scope=scope,times=times,statistic=stat,evidence_ids=ids,formula=path,exact_fraction=str(value),unit='degC'))
def engine(c,method):
 answers=[]
 for t in c['targets']:
  if t['statistic']=='difference':
   a,b=answers[:2]; good=a['kind']==b['kind']=='direct'; answers.append(dict(kind='direct' if good else 'unresolved',value=b['value']-a['value'] if good else None)); continue
  if method=='H_naive_visible':
   values=[v for e in c['evidence'] if e['type']=='raw' and e['scope']==t['scope'] for v in e['values']]; answers.append(dict(kind='direct' if values else 'unresolved',value=(float(np.mean(values)) if t['statistic']=='mean' else max(values)) if values else None)); continue
  answers.append(solve(t['scope'],t['times'],t['statistic'],c['evidence'],method=='A_raw_only',method=='A_scope_off'))
 if method=='A_all_or_none' and any(a['kind']!='direct' for a in answers): answers=[dict(kind='unresolved',value=None) for a in answers]
 return answers
def run():
 public=json.loads((ROOT/'inputs/E1_public.json').read_text(encoding='utf-8')); gold=json.loads((ROOT/'inputs/E1_gold.json').read_text(encoding='utf-8')); groups=pd.read_csv(ROOT/'inputs/E1_groups.csv'); methods=['M_exact','H_naive_visible','A_raw_only','A_scope_off','A_all_or_none']; rows=[]; certificates=[]
 # Same inherited benchmark held fixed for comparability; no claim of a new blind benchmark.
 for c,g,meta in zip(public,gold,groups.to_dict('records')):
  for method in methods:
   answers=engine(c,method)
   for i,(a,t) in enumerate(zip(answers,g['targets'])):
    rows.append(dict(**meta,method=method,target=i,gold_direct=t['direct'],delivered_direct=a['kind']=='direct',numeric_correct=a['kind']=='direct' and t['direct'] and abs(a['value']-t['value'])<.0050001,value=a['value'],gold_value=t['value']))
    if method=='M_exact' and a.get('certificate'): certificates.append(dict(case_id=c['case_id'],target=i,certificate=a['certificate']))
 z=pd.DataFrame(rows); z.to_csv(OUT/'evidence_predictions.csv',index=False); (OUT/'certificates.json').write_text(json.dumps(certificates,ensure_ascii=False,indent=2),encoding='utf-8'); metrics=[]
 for (split,method),g in z.groupby(['split','method']):
  neg=g[~g.gold_direct]; pos=g[g.gold_direct]; direct=g[g.delivered_direct]; metrics.append(dict(split=split,method=method,n_targets=len(g),unsupported_n=int(neg.delivered_direct.sum()),unsupported_den=len(neg),UAR=float(neg.delivered_direct.mean()),false_refusal_n=int((~pos.delivered_direct).sum()),false_refusal_den=len(pos),FAR=float((~pos.delivered_direct).mean()),coverage=float(g.delivered_direct.mean()),direct_accuracy=float(direct.numeric_correct.mean())))
 pd.DataFrame(metrics).to_csv(OUT/'evidence_metrics.csv',index=False)
 # Explicitly authored independent-of-generator challenge cases; authored by this assistant, not human labels.
 T=[0,6,12,18]
 def raw(v,ts=T,scope='A',id='r',unit='degC'): return dict(id=id,scope=scope,times=ts,type='raw',values=v,unit=unit)
 cases=[
  ('complete',T,'mean',[raw([20,22,24,26])],'direct',23),
  ('missing',T,'mean',[raw([20,22],[0,6])],'unresolved',None),
  ('sum_mean',T,'mean',[dict(id='s',scope='A',times=T,type='sum',value=92,n=4)],'direct',23),
  ('sum_not_max',T,'max',[dict(id='s',scope='A',times=T,type='sum',value=92,n=4)],'unresolved',None),
  ('unequal_groups',T,'mean',[dict(id='g1',scope='A',times=[0],type='group_mean',n=1,value=20),dict(id='g2',scope='A',times=[6,12,18],type='group_mean',n=3,value=24)],'direct',23),
  ('overlap',T,'mean',[dict(id='g1',scope='A',times=[0,6],type='group_mean',n=2,value=21),dict(id='g2',scope='A',times=[6,12],type='group_mean',n=2,value=23)],'unresolved',None),
  ('wrong_station',T,'mean',[raw([20,22,24,26],scope='B')],'unresolved',None),
  ('local_contract',[0,6],'mean',[raw([20,22],[0,6])],'direct',21),
  ('duplicates',T,'mean',[raw([20,22,24,26]),raw([20],[0],id='dup')],'direct',23),
  ('raw_conflict',T,'mean',[raw([20,22,24,26]),raw([21],[0],id='bad')],'conflict',None),
  ('summary_conflict',T,'mean',[raw([20,22,24,26]),dict(id='s',scope='A',times=T,type='mean',value=24)],'conflict',None),
  ('fahrenheit',T,'mean',[raw([68,71.6,75.2,78.8],unit='degF')],'direct',23),
 ]
 challenges=[]
 for name,ts,stat,es,kind,value in cases:
  a=solve('A',ts,stat,es); passed=a['kind']==kind and (value is None or abs(a['value']-value)<.0050001); challenges.append(dict(name=name,times=ts,statistic=stat,evidence=es,expected_kind=kind,expected_value=value,actual=a,passed=passed,author='current assistant, not independent human review'))
 (OUT/'challenge_cases.json').write_text(json.dumps(challenges,ensure_ascii=False,indent=2),encoding='utf-8'); assert all(x['passed'] for x in challenges)
 # Insufficiency witnesses share the visible prefix but yield distinct target results.
 witnesses=[dict(visible=[20,22],completion_a=[20,22,24,26],completion_b=[20,22,28,30],mean_a=23,mean_b=25),dict(sum=92,n=4,completion_a=[20,22,24,26],completion_b=[21,23,23,25],max_a=26,max_b=25)]
 (OUT/'insufficiency_witnesses.json').write_text(json.dumps(witnesses,indent=2),encoding='utf-8')
 # Cluster bootstrap by original base group, jointly retaining targets and variants.
 boot=[]; rng=np.random.default_rng(20261113)
 for method,g in z[z.split=='test'].groupby('method'):
  blocks=[h for _,h in g.groupby('base_id')]; draws=[]
  for _ in range(2000):
   a=pd.concat([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]); neg=a[~a.gold_direct]; pos=a[a.gold_direct]; draws.append([neg.delivered_direct.mean(),(~pos.delivered_direct).mean(),a.delivered_direct.mean()])
  d=np.array(draws)
  for j,metric in enumerate(['UAR','FAR','coverage']): boot.append(dict(method=method,metric=metric,lower=float(np.quantile(d[:,j],.025)),upper=float(np.quantile(d[:,j],.975)),bootstrap_clusters=30,n_boot=2000))
 pd.DataFrame(boot).to_csv(OUT/'evidence_bootstrap.csv',index=False)
 stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'legacy_tests'))); (OUT/'software_tests.txt').write_text(stream.getvalue(),encoding='utf-8'); assert result.wasSuccessful()
 print('E1',len(public),'cases, challenges',len(challenges),'passed; legacy regression tests',result.testsRun,flush=True)
if __name__=='__main__': run()
