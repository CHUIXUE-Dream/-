from pathlib import Path
import sys,os,json,hashlib,datetime,gzip,urllib.request,math,sqlite3,ast
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P/'.deps'),str(P/'v5')]
import run_experiment as e
import numpy as np,pandas as pd
D=R/'data';O=R/'results';D.mkdir(exist_ok=True);O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def hobj(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def stages(c,reference,policy,p0):
 day=pd.Timestamp(c['date'],tz='UTC');anchor=c['last_visible_value'];at=(pd.Timestamp(c['last_visible_time'])-day).total_seconds()/3600 if anchor is not None else -24
 known={};unseen=list(range(4));order=[];out=[]
 for b in range(5):
  xs=([at] if anchor is not None else [])+[i*6 for i in known];vs=([anchor] if anchor is not None else [])+[known[i] for i in known]
  sort=np.argsort(xs);pred=float(np.mean(np.interp([0,6,12,18],np.array(xs)[sort],np.array(vs)[sort]))) if xs else np.nan
  if b==0 and policy!='L_only':pred=p0
  out.append(dict(budget=b,prediction=pred,revealed_hours=[i*6 for i in order]))
  if not unseen:break
  if policy.startswith(('T','all')):chosen=max(unseen,key=lambda i:(min(abs(i*6-t) for t in xs) if xs else 999,-i))
  elif policy.startswith('H'):chosen=min(unseen,key=lambda i:hashlib.sha256(f"{c['case_id']}|{i}".encode()).hexdigest())
  else:chosen=min(unseen)
  known[chosen]=reference[chosen];order.append(chosen);unseen.remove(chosen)
 return out
POLICIES=['B_only','L_only','T_fixed2','H_adapt2','E_adapt2','T_adapt2','T_adapt4','all4']
bundle=e.load_bundle()
if not (O/'frozen.json').exists():
 y,ids=e.old.prep();cases,gold=e.old.build(y,ids,2024);gd=gold.set_index('case_id');cases=[c for c in cases if gd.loc[c['case_id'],'scenario']=='mask4'];rows=[]
 for c in cases:
  st=e.old.state(c,bundle['base']);p0=e.old_values(c,st,'mean',bundle['regs'],bundle['routes'])['v4'];g=gd.loc[c['case_id']]
  for pol in POLICIES:
   for a in stages(c,g.reference_values,pol,p0):rows.append(dict(station=c['station'],policy=pol,budget=a['budget'],error=a['prediction']-g.reference_mean))
 calrows=pd.DataFrame(rows);cal={}
 for (s,pol,b),g in calrows.groupby(['station','policy','budget']):
  n=len(g);k=math.ceil((n+1)*.9);q=0. if b==4 else float(np.sort(g.error.abs())[k-1]) if n>=100 and k<=n else None;cal[f'{s}|{pol}|{b}']=dict(n=n,q=q)
 calrows.to_csv(O/'calibration_errors.csv.gz',index=False);dump(O/'calibration.json',cal)
 dump(O/'frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=sha(__file__),protocol_sha256=sha(R/'PROTOCOL.md'),calibration_sha256=sha(O/'calibration.json'),base_model_sha256=sha(P/'v5/results/models.pkl'),year=2020,primary_tau=.25))
freeze=json.loads((O/'frozen.json').read_text());assert sha(__file__)==freeze['code_sha256'];assert sha(O/'calibration.json')==freeze['calibration_sha256']
raw=D/'raw';raw.mkdir(exist_ok=True);manifest=[];lines=[]
for s in e.STATIONS:
 path=raw/f'{s}h2020.txt.gz';url=f'https://www.ndbc.noaa.gov/data/historical/stdmet/{s}h2020.txt.gz'
 if path.exists():
  b=path.read_bytes();prior=json.loads((D/'source_manifest.json').read_text());retrieved=next(a['retrieved_utc'] for a in prior if a['station']==s)
 else:
  b=urllib.request.urlopen(url,timeout=45).read();gzip.decompress(b);path.write_bytes(b);retrieved=datetime.datetime.now(datetime.timezone.utc).isoformat()
 digest=hashlib.sha256(b).hexdigest();manifest.append(dict(station=s,year=2020,url=url,retrieved_utc=retrieved,sha256=digest,bytes=len(b)));dump(D/'source_manifest.json',manifest)
 text=gzip.decompress(b).decode('ascii');head=text.splitlines()[0].lstrip('#').split();wi=head.index('WTMP')
 for ln,line in enumerate(text.splitlines(),1):
  if line.startswith('#') or not line.strip():continue
  a=line.split()
  try:
   v=float(a[wi]);assert -5<=v<=45;t=pd.Timestamp(year=int(a[0]),month=int(a[1]),day=int(a[2]),hour=int(a[3]),minute=int(a[4]),tz='UTC');lines.append(dict(station=s,time=t,hour=t.floor('h'),value=v,record_id=f'v7:{s}:{ln}',source_sha256=digest))
  except (ValueError,AssertionError):continue
f=pd.DataFrame(lines).sort_values('time').drop_duplicates(['station','hour'],keep='last');f.to_csv(D/'hourly_lineage.csv.gz',index=False);ix=pd.date_range('2020-01-01','2021-01-01',freq='h',tz='UTC',inclusive='left');y=f.pivot(index='hour',columns='station',values='value').reindex(index=ix,columns=e.STATIONS);ids=f.pivot(index='hour',columns='station',values='record_id').reindex(index=ix,columns=e.STATIONS)
cases,gold=e.build_snapshot(y,ids);gold=gold[gold.scenario=='mask4'].copy();keep=set(gold.case_id);cases=[c for c in cases if c['case_id'] in keep];dump(D/'public.json',cases);gold.to_json(D/'gold.json',orient='records',indent=2);gd=gold.set_index('case_id')
records=[]
for c in cases:
 st=e.old.state(c,bundle['base']);x,od=e.features(c,st);records.append((c,st,x,od))
forecast=e.forecast(records,bundle);forecast.to_csv(O/'forecasts_without_gold.csv.gz',index=False);z=e.attach_gold(forecast,gold);z.to_csv(O/'point_predictions.csv.gz',index=False);base=z[(z.method=='v4')&(z.statistic=='mean')].set_index('case_id');cal=json.loads((O/'calibration.json').read_text());events=[];trace=[]
for c in cases:
 g=gd.loc[c['case_id']]
 for pol in POLICIES:
  path=stages(c,g.reference_values,pol,float(base.loc[c['case_id'],'prediction']))
  for tau in [.1,.25,.5]:
   cap=0 if pol.endswith('only') else 2 if pol.endswith('2') else 4
   for a in path[:cap+1]:
    b=a['budget'];q=cal[f"{c['station']}|{pol}|{b}"]['q'];delivered=b==4 or (q is not None and q<=tau)
    if pol in ['T_fixed2','all4'] and b<cap:continue
    if delivered or b==cap:break
   err=a['prediction']-g.reference_mean;kind='direct' if b==4 else 'estimate' if delivered else 'unresolved';row=dict(case_id=c['case_id'],station=c['station'],date=c['date'],policy=pol,tau=tau,budget=b,prediction=a['prediction'],reference=g.reference_mean,error=err,q=q,delivered=delivered,kind=kind,qualified=bool(delivered and abs(err)<=tau),covered=bool(q is not None and abs(err)<=q),revealed_hours=','.join(map(str,a['revealed_hours'])))
   events.append(row)
ev=pd.DataFrame(events);ev.to_csv(O/'workflow_events.csv.gz',index=False);summary=[]
for (pol,tau),g in ev.groupby(['policy','tau']):
 delivered=g[g.delivered];est=delivered[delivered.kind=='estimate'];summary.append(dict(policy=pol,tau=tau,n=len(g),delivered=len(delivered),delivery_rate=g.delivered.mean(),qualified_rate=g.qualified.mean(),avg_records=g.budget.mean(),direct=int(g.kind.eq('direct').sum()),selected_PICP=float(est.covered.mean()) if len(est) else None,delivered_RMSE=e.rmse(delivered.error) if len(delivered) else None))
pd.DataFrame(summary).to_csv(O/'workflow_metrics.csv',index=False)
# Integrate exact-key path/result reuse with the same safe LRU implementation; no latency claim.
source=(P/'v3/cache_experiment.py').read_text(encoding='utf-8');node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Cache');exec(ast.get_source_segment(source,node),globals());cache=Cache('A_no_hotness');hits=0;plans=0;unsafe=0;i=0;claims=[];audits=[]
selected=ev[(ev.policy=='T_adapt4')&(ev.tau==.25)].set_index('case_id')
for c in cases:
 r=selected.loc[c['case_id']];contract=dict(case=c['case_id'],station=c['station'],hours=c['hours'],stat='mean',tau=.25,model=freeze['base_model_sha256'],calibration=freeze['calibration_sha256'],rule=freeze['code_sha256'],data=[m['sha256'] for m in manifest],mode='estimate_allowed');k=hobj(contract);pk=hobj({a:contract[a] for a in ['station','stat','tau','model','calibration','rule','mode']});ans=dict(value=float(r.prediction) if r.delivered else None,kind=r.kind,budget=int(r.budget),q=float(r.q) if pd.notna(r.q) else None)
 for repeat in range(2):
  got=cache.get('r'+k,i);hits+=got is not None
  if got is None:
   plan=cache.get('p'+pk,i);plans+=plan is not None
   if plan is None:cache.put('p'+pk,'path',dict(policy='T_adapt4',kind_floor='estimate'),1,i)
   got=ans;cache.put('r'+k,'result',got,1,i)
  unsafe+=got!=ans;i+=1
 claim=dict(claim_id=c['case_id'],**ans,contract_hash=k);claims.append(claim);audits.append(dict(parent=c['case_id'],reference=float(gd.loc[c['case_id']].reference_mean),sources=gd.loc[c['case_id']].reference_sources,restored_kind='direct',restored_value=float(gd.loc[c['case_id']].reference_mean)))
dump(O/'claims.json',claims);before=sha(O/'claims.json');dump(O/'restored_certificates.json',audits);assert before==sha(O/'claims.json')
resolved=0;source_ok=True
rawlines={s:gzip.decompress((raw/f'{s}h2020.txt.gz').read_bytes()).decode('ascii').splitlines() for s in e.STATIONS}
for g in gold.itertuples():
 vals=[]
 for item in g.reference_sources:
  _,s,ln=item.split(':');hd=rawlines[s][0].lstrip('#').split();vals.append(float(rawlines[s][int(ln)-1].split()[hd.index('WTMP')]));resolved+=1
 source_ok &= math.isclose(sum(vals)/4,g.reference_mean,abs_tol=1e-12)
checks=dict(code_frozen=sha(__file__)==freeze['code_sha256'],calibration_frozen=sha(O/'calibration.json')==freeze['calibration_sha256'],retrieved_after_freeze=all(m['retrieved_utc']>freeze['created_utc'] for m in manifest),same_denominators=ev.groupby(['policy','tau']).size().eq(len(cases)).all().item(),source_rows_resolve=source_ok,safe_cache_no_mismatch=unsafe==0,all4_exact=ev[ev.policy=='all4'].error.abs().le(1e-12).all().item(),claims_preserved=before==sha(O/'claims.json'),tau_gate=ev[ev.delivered&ev.kind.eq('estimate')].apply(lambda r:r.q<=r.tau,axis=1).all().item())
dump(O/'audit.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),requests=len(cases),by_station=gold.groupby('station').size().to_dict(),source_resolutions=resolved,result_hits=hits,path_hits=plans,replayed_requests=i,cache_class_sha256=hobj(ast.get_source_segment(source,node))))
assert all(checks.values()),checks
print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(checks))
