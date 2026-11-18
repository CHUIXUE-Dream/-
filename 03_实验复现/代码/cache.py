"""Real numerical path execution, bounded SQLite result/plan caches and adversarial probes."""
from pathlib import Path
import sys,os,json,hashlib,time,sqlite3,copy,math,collections
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent/'.deps'))
import numpy as np,pandas as pd
from experiment import infer,seasonal_at
OUT=ROOT/'results'
def hashobj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
models=json.loads((OUT/'models.json').read_text(encoding='utf-8')); metadata={x['id']:x for x in json.loads((ROOT/'data/stations.json').read_text(encoding='utf-8'))}
cal=pd.read_csv(OUT/'predictions.csv.gz',dtype={'station':str}); cal=cal[(cal.split=='cal')&(cal.method=='M_est')&(cal.statistic=='mean')]
calmap={ (r.station,r.pattern):(r.q,r.n_cal) for r in cal[cal.scenario=='mask4'].itertuples() if pd.notna(r.q)}
MODEL=hashobj(models); CAL=hashobj({str(k):v for k,v in calmap.items()})
def plan(c):
 s=c['station']; steps=[]
 for i in range(4):
  if c['target_visible'][i] is not None: steps.append({'observed':True}); continue
  es=[e for j,e in models[s]['edges'].items() if e['eligible'] and c['donors'].get(j,[None]*4)[i] is not None]
  es=sorted(es,key=lambda e:(-max(e['rho'],0)**2/(e['dev_rmse']**2+1e-6),e['donor']))
  if not es: return None
  w=np.array([max(e['rho'],0)**2/(e['dev_rmse']**2+1e-6) for e in es]); w/=w.sum(); steps.append(dict(edges=[dict(donor=e['donor'],b=e['b'],w=float(v)) for e,v in zip(es,w)]))
 return dict(station=s,steps=steps,model=MODEL,calibration=CAL,rule='v3.0',kind_floor='estimate')
def execute(c,p):
 if p is None: return dict(kind='hypothesis',value=None,pattern='unavailable')
 s=c['station']; values=[]; signature=[]; ood=False
 for i,step in enumerate(p['steps']):
  if step.get('observed'): values.append(c['target_visible'][i]); signature.append('observed'); continue
  t=pd.Timestamp(c['hours'][i]); base=seasonal_at(models[s],t); val=0; donors=[]
  for e in step['edges']:
   j=e['donor']; x=c['donors'][j][i]; bounds=models[s]['bounds'][j]; ood|=not(bounds[0]<=x<=bounds[1]); val+=e['w']*(base+e['b'][0]+e['b'][1]*(x-seasonal_at(models[j],t))); donors.append(j)
  values.append(val); signature.append(','.join(donors))
 pattern=';'.join(signature); q,n=calmap.get((s,pattern),(None,0)); eligible=q is not None and not ood
 return dict(kind='estimate' if eligible else 'hypothesis',value=float(np.mean(values)) if eligible else None,candidate=float(np.mean(values)),q=q,n_cal=int(n),pattern=pattern)
def reference(c):
 r=infer(c,models,metadata,'M_est')
 if r is None: return dict(kind='hypothesis',value=None,pattern='unavailable')
 q,n=calmap.get((c['station'],r['pattern']),(None,0)); eligible=q is not None and not r['out_of_domain']
 return dict(kind='estimate' if eligible else 'hypothesis',value=r['mean'] if eligible else None,candidate=r['mean'],q=q,n_cal=int(n),pattern=r['pattern'])
def same(a,b):
 return a['kind']==b['kind'] and ((a.get('value') is None and b.get('value') is None) or (a.get('value') is not None and b.get('value') is not None and abs(a['value']-b['value'])<1e-9)) and a.get('pattern')==b.get('pattern')
def key(c,version='snapshot1',model=MODEL,cal=CAL,mode='estimate_allowed',unit='degC',statistic='mean'):
 return hashobj(dict(station=c['station'],variable='WTMP',times=c['hours'],sample='four-hour-representatives',answer_mode=mode,unit=unit,statistic=statistic,mask_world=c['case_id'],data_version=version,model=model,calibration=cal,rule='v3.0'))
def plankey(c):
 return hashobj(dict(station=c['station'],available={j:[v is not None for v in vs] for j,vs in c['donors'].items()},observed=[v is not None for v in c['target_visible']],model=MODEL,calibration=CAL,rule='v3.0',statistic='mean'))
class Cache:
 def __init__(self,policy):
  self.policy=policy; self.db=sqlite3.connect(':memory:'); self.db.execute('CREATE TABLE entries(key TEXT PRIMARY KEY, layer TEXT,payload TEXT,size INTEGER,last_access INTEGER,frequency REAL,cost REAL)'); self.evictions=0
 def get(self,k,i):
  r=self.db.execute('SELECT payload,last_access,frequency FROM entries WHERE key=?',(k,)).fetchone()
  if not r: return None
  f=r[2]*2**(-(i-r[1])/100)+1; self.db.execute('UPDATE entries SET last_access=?,frequency=? WHERE key=?',(i,f,k)); return json.loads(r[0])
 def put(self,k,layer,value,cost,i):
  payload=json.dumps(value,sort_keys=True); size=len(payload.encode())+len(k)+64
  self.db.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?)',(k,layer,payload,size,i,1,float(cost)))
  while True:
   n,total=self.db.execute('SELECT COUNT(*),COALESCE(SUM(size),0) FROM entries').fetchone()
   if n<=128 and total<=262144: break
   rows=self.db.execute('SELECT key,size,last_access,frequency,cost FROM entries').fetchall()
   if self.policy in ['C1_LRU','A_no_hotness']: victim=min(rows,key=lambda r:r[2])[0]
   else: victim=min(rows,key=lambda r:r[3]*2**(-(i-r[2])/100)*r[4]/(r[1]+1))[0]
   self.db.execute('DELETE FROM entries WHERE key=?',(victim,)); self.evictions+=1
  if i%100==0 and self.policy not in ['C1_LRU','A_no_hotness']:
   for k,last,f in self.db.execute('SELECT key,last_access,frequency FROM entries').fetchall(): self.db.execute('UPDATE entries SET frequency=?,last_access=? WHERE key=?',(f*2**(-(i-last)/100),i,k))
  self.db.commit()
def run():
 gold=pd.read_csv(OUT/'case_registry.csv',dtype={'station':str}); mask4=set(gold.loc[(gold.split=='test')&(gold.scenario=='mask4'),'case_id']); pool=[json.loads(l) for l in (ROOT/'data/public/test.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(l)['case_id'] in mask4]
 assert pool
 events=[]; summary=[]; policies=['C0_resident_no_cache','C1_LRU','C2_naive_similar','C3_hot_path','A_result_only','A_no_hotness','A_no_version','A_no_kind']
 for seed in [20261112,20261113,20261114]:
  rng=np.random.default_rng(seed); requests=[]
  for i in range(1000):
   if i%10<3 and requests: c=copy.deepcopy(requests[int(rng.integers(0,len(requests)))][0])
   else:
    hot=['41009','41010'] if i<400 else ['41010','41013']; candidates=[c for c in pool if c['station'] in hot] if rng.random()<.7 else pool; c=copy.deepcopy(candidates[int(rng.integers(0,len(candidates)))])
   # 30/40/30 categories: exact repeats / template new dates / combinations.
   if 3<=i%10<7:
    c['case_id']=hashobj(dict(case=c['case_id'],request=i)); c['hours']=[str(pd.Timestamp(h)+pd.Timedelta(hours=24*(i+1))) for h in c['hours']]
    # Synthetic query uses unchanged donor values at synthetic dates; cache quality only, not ocean accuracy.
   version='snapshot1' if i<700 else 'snapshot2'
   if i>=700 and int(hashobj(c['case_id'])[:8],16)%10==0 and not c.get('_snapshot2_applied',False):
    c['donors']={j:[None if v is None else v+.2 for v in vs] for j,vs in c['donors'].items()}
    c['_snapshot2_applied']=True
   requests.append((c,version))
  truths=[reference(c) for c,v in requests]
  for repeat in range(5):
   for policy in policies:
    cache=Cache(policy); lat=[]; hits=0; phits=0; unsafe=0; reads=0
    for i,((c,v),truth) in enumerate(zip(requests,truths)):
     begin=time.perf_counter_ns(); k=key(c,'ignored' if policy=='A_no_version' else v); pk=plankey(c); result_hit=False; path_hit=False
     if policy=='C2_naive_similar': k=c['station']+'|mean'
     ans=None if policy=='C0_resident_no_cache' else cache.get('r'+k,i)
     if ans is not None: result_hit=True
     elif policy in ['C3_hot_path','A_no_hotness','A_no_version','A_no_kind']:
      p=cache.get('p'+pk,i); path_hit=p is not None
      if p is None:
       a=time.perf_counter_ns(); p=plan(c); cost=time.perf_counter_ns()-a
       if p is not None: cache.put('p'+pk,'plan',p,cost,i)
      ans=execute(c,p)
     else: ans=reference(c)
     if policy=='A_no_kind' and result_hit and ans['kind']=='estimate': ans=dict(ans,kind='direct')
     if not result_hit and policy!='C0_resident_no_cache': cache.put('r'+k,'result',ans,time.perf_counter_ns()-begin,i)
     ms=(time.perf_counter_ns()-begin)/1e6; equal=same(ans,truth); hits+=result_hit; phits+=path_hit; unsafe+=not equal; lat.append(ms)
     events.append(dict(seed=seed,repeat=repeat,policy=policy,request=i,result_hit=result_hit,path_hit=path_hit,latency_ms=ms,unsafe=not equal,stage='cold' if i<100 else 'stable' if i<400 else 'migration' if i<700 else 'updated'))
    count,bytes_=cache.db.execute('SELECT count(*),COALESCE(sum(size),0) FROM entries').fetchone(); summary.append(dict(seed=seed,repeat=repeat,policy=policy,n=1000,result_hit_rate=hits/1000,path_hit_rate=phits/1000,unsafe_rate=unsafe/1000,P50_ms=float(np.median(lat)),P95_ms=float(np.quantile(lat,.95)),evictions=cache.evictions,entries=count,cache_payload_bytes=bytes_))
    if repeat==0:
     cache.db.commit()
     target=sqlite3.connect(OUT/f'cache_{seed}_{policy}.sqlite'); cache.db.backup(target); target.close()
    cache.db.close()
  print('E3 seed',seed,'done',flush=True)
 pd.DataFrame(events).to_csv(OUT/'cache_events.csv.gz',index=False); pd.DataFrame(summary).to_csv(OUT/'cache_metrics.csv',index=False)
 # Explicit near collisions: exact scope/version/unit/statistic/mode/model/calibration change.
 probes=[]
 for i in range(40):
  c=copy.deepcopy(pool[i%len(pool)]); old=key(c); dimension=['date','version','unit','statistic','mode','model','calibration','station'][i%8]; args={}
  if dimension=='date': c['hours']=[str(pd.Timestamp(h)+pd.Timedelta(days=1)) for h in c['hours']]
  elif dimension=='station': c['station']=next(s for s in models if s!=c['station'])
  else: args[dict(version='version',unit='unit',statistic='statistic',mode='mode',model='model',calibration='cal')[''+dimension]]=dict(version='snapshot2',unit='degF',statistic='max',mode='exact_only',model='new_model',cal='new_calibration')[dict(calibration='cal').get(dimension,dimension)]
  new=key(c,**args); probes.append(dict(probe=i,dimension=dimension,old_key=old,new_key=new,secure_result_miss=old!=new))
 pd.DataFrame(probes).to_csv(OUT/'cache_collision_probes.csv',index=False)
 assert all(x['secure_result_miss'] for x in probes)
 safe=pd.DataFrame(summary); assert safe[safe.policy.isin(['C1_LRU','C3_hot_path','A_result_only','A_no_hotness'])].unsafe_rate.eq(0).all()
 print(safe.groupby('policy')[['P50_ms','P95_ms','unsafe_rate','result_hit_rate','path_hit_rate']].median().to_string())
if __name__=='__main__': run()
