"""Reproducible v5: train/dev only selection, freeze before realtime retrieval."""
from pathlib import Path
import os, sys, json, hashlib, math, datetime, urllib.request, pickle, argparse
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
R=Path(__file__).resolve().parent; P=R.parent
sys.path[:0]=[str(P/'.deps'),str(P/'v4')]
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import optimize as old
O=R/'results'; D=R/'data'; O.mkdir(exist_ok=True); D.mkdir(exist_ok=True)
STATIONS=['41009','41010','41013']; SCENARIOS=old.SCENARIOS
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
def rmse(e): return float(np.sqrt(np.mean(np.square(e))))
def ctx(c,st,stat): return f"{c['station']}|{st['mask_count']}|{st['mode']}|{stat}"
def features(c,st):
    anchor=c['last_visible_value']; anchor=anchor if anchor is not None else 0.
    visible=c['target_visible']; vals=[v for v in visible if v is not None]
    x=list(st['x'])+[v-anchor if v is not None else 0 for v in visible]+[int(v is not None) for v in visible]
    x += [float(np.mean(vals)-anchor) if vals else 0., float(np.ptp(vals)) if vals else 0.]
    changes=[]
    for j in sorted(c['donors']):
        vs=c['donors'][j]; a=c['donor_anchor'][j]
        ds=[v-a if v is not None and a is not None else 0. for v in vs]
        x+=ds+[int(v is not None and a is not None) for v in vs]
        changes += [abs(v-a) for v in vs if v is not None and a is not None]
    return np.array(x),max(changes) if changes else 999.
def old_values(c,st,stat,regs,routes):
    cand=old.candidates(st,stat,regs[c['station']+'|'+stat]); route=routes.get(ctx(c,st,stat),{'method':'P_local_interp','alpha':1})
    target=cand.get(route['method']); loc=cand['P_local_interp']; value=loc if target is None else loc*(1-route['alpha'])+target*route['alpha']
    return {'local':loc,'persistence':cand['P_persistence'] if cand['P_persistence'] is not None else loc,'v4':value}
def historical():
    y,ids=old.prep(); base=json.loads((P/'v3/results/models.json').read_text(encoding='utf-8'))
    sets={}
    for year in [2022,2023,2024]:
        cases,gold=old.build(y,ids,year); cases=[c for c in cases if gold.set_index('case_id').loc[c['case_id'],'scenario']!='donor_missing'] if False else cases
        gd=gold.set_index('case_id'); cases=[c for c in cases if gd.loc[c['case_id'],'scenario'] in ['mask1','mask2','mask4','block72']]
        records=[]
        for c in cases:
            st=old.state(c,base); x,ood=features(c,st)
            records.append((c,st,x,ood))
        sets[year]=(records,gd)
        print('historical prepared',year,len(records),flush=True)
    return sets,base
def fit_phase():
    if (O/'frozen.json').exists(): raise SystemExit('Freeze exists; use --phase evaluate or a new version. No overwriting freeze.')
    sets,base=historical(); regs=json.loads((P/'v4/results/regressions.json').read_text(encoding='utf-8')); routes=json.loads((P/'v4/results/routes.json').read_text(encoding='utf-8'))
    models={}; decisions={}; rows=[]
    for s in STATIONS:
        for stat in ['mean','max']:
            train=[r for r in sets[2022][0] if r[0]['station']==s]; dev=[r for r in sets[2023][0] if r[0]['station']==s]
            X=np.array([r[2] for r in train]); Z=np.array([r[2] for r in dev]); locals_=np.array([old_values(c,st,stat,regs,routes)['local'] for c,st,x,od in train]); yy=np.array([sets[2022][1].loc[c['case_id'],'reference_'+stat] for c,st,x,od in train])-locals_
            fitted={}
            for alpha in [10,100]: fitted['ridge'+str(alpha)]=old.fit(X,yy,alpha)
            for leaves in [5,9]:
                model=HistGradientBoostingRegressor(max_leaf_nodes=leaves,max_iter=150,learning_rate=.05,min_samples_leaf=30,l2_regularization=10,early_stopping=False,random_state=20261113)
                fitted['tree'+str(leaves)]=model.fit(X,yy)
            for q in [.05,.95]:
                model=HistGradientBoostingRegressor(loss='quantile',quantile=q,max_leaf_nodes=5,max_iter=150,learning_rate=.05,min_samples_leaf=30,l2_regularization=10,early_stopping=False,random_state=20261113)
                fitted['q'+str(q)]=model.fit(X,yy)
            models[s+'|'+stat]=fitted
            predictions={name:np.array([old.pred(m,x) for x in Z]) if name.startswith('ridge') else m.predict(Z) for name,m in fitted.items() if not name.startswith('q')}
            frame=[]
            for i,(c,st,x,od) in enumerate(dev):
                vv=old_values(c,st,stat,regs,routes); frame.append(dict(context=ctx(c,st,stat),reference=sets[2023][1].loc[c['case_id'],'reference_'+stat],ood=od,**vv,**{name:vv['local']+v[i] for name,v in predictions.items()}))
            f=pd.DataFrame(frame)
            for context,g in f.groupby('context'):
                best='v4'; b=rmse(g.v4-g.reference); bma=float(abs(g.v4-g.reference).mean())
                for name in predictions:
                    r=rmse(g[name]-g.reference); ma=float(abs(g[name]-g.reference).mean())
                    if len(g)>=30 and r<b*.98 and ma<=bma*1.02: best=name;b=r;bma=ma
                finite=g.ood[g.ood<999]; threshold=max(.5,float(finite.quantile(.99))) if len(finite) else .5
                decisions[context]=dict(method=best,threshold=threshold,n_dev=len(g))
                rows.append(dict(context=context,selected=best,n_dev=len(g),v4_rmse=rmse(g.v4-g.reference),selected_rmse=b,threshold=threshold))
    (O/'models.pkl').write_bytes(pickle.dumps(models,protocol=5)); dump(O/'decisions.json',decisions);pd.DataFrame(rows).to_csv(O/'dev_selection.csv',index=False)
    bundle=dict(models=models,decisions=decisions,base=base,regs=regs,routes=routes)
    # Generate predictions from public information before independent gold merge.
    cal=forecast(sets[2024][0],bundle)
    cal.to_csv(O/'cal_forecasts_without_gold.csv.gz',index=False)
    cal=attach_gold(cal,sets[2024][1]); calib={}
    for key,g in cal.groupby(['context','method']):
        # One observation per date in this group; scenario duplicates never used.
        g=g.drop_duplicates(['station','date']);n=len(g);k=math.ceil((n+1)*.9)
        scores=np.maximum.reduce([g.lower_raw-g.reference,g.reference-g.upper_raw,np.zeros(n)])
        q=float(np.sort(scores)[k-1]) if n>=100 and k<=n else None
        ae=abs(g.prediction-g.reference).to_numpy(); qfixed=float(np.sort(ae)[k-1]) if n>=100 and k<=n else None
        calib['|'.join(key)]={'n':n,'q':q,'q_fixed':qfixed}
    dump(O/'calibration.json',calib)
    cal.to_csv(O/'cal_predictions.csv.gz',index=False)
    freeze=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),version='v5',code_sha256=sha(__file__),protocol_sha256=sha(R/'PROTOCOL.md'),model_sha256=sha(O/'models.pkl'),decisions_sha256=sha(O/'decisions.json'),calibration_sha256=sha(O/'calibration.json'),historical_manifest_sha256=sha(P/'v3/data/manifest.json'),v4_freeze_sha256=sha(P/'v4/results/frozen.json'),evaluation='Not yet downloaded. First locked realtime snapshot will be evaluated once; no tuning on it.',seed=20261113)
    dump(O/'frozen.json',freeze); print('FROZEN before any realtime data access',flush=True)
def forecast(records,bundle):
    out=[];models=bundle['models']; decisions=bundle['decisions']; regs=bundle['regs'];routes=bundle['routes']
    # Batch trees to avoid one-sample estimator overhead.
    for s in STATIONS:
        subset=[r for r in records if r[0]['station']==s]
        if not subset:continue
        X=np.array([r[2] for r in subset])
        for stat in ['mean','max']:
            fitted=models[s+'|'+stat]; vv={name:np.array([old.pred(m,x) for x in X]) if name.startswith('ridge') else m.predict(X) for name,m in fitted.items()}
            for i,(c,st,x,od) in enumerate(subset):
                values=old_values(c,st,stat,regs,routes);context=ctx(c,st,stat);rule=decisions.get(context,dict(method='v4',threshold=.5));chosen=rule['method'];noguard=values['v4'] if chosen=='v4' else values['local']+vv[chosen][i];flag=od>rule['threshold'] or st['donor_count']==0
                values['v5_no_guard']=noguard;values['v5']=values['local'] if flag else noguard
                a=values['local']+vv['q0.05'][i];b=values['local']+vv['q0.95'][i];lo=min(a,b);hi=max(a,b)
                for method,prediction in values.items():
                    # CQR bounds for each method are recentered to that method's point.
                    lower=prediction+(lo-values['local']);upper=prediction+(hi-values['local'])
                    out.append(dict(case_id=c['case_id'],station=s,date=c['date'],scenario=c.get('scenario','normal'),statistic=stat,context=context,method=method,prediction=float(prediction),lower_raw=float(lower),upper_raw=float(upper),ood_score=float(od),ood_threshold=rule['threshold'],fallback=bool(flag) if method=='v5' else False,selected=chosen,mask_count=st['mask_count']))
    return pd.DataFrame(out)
def attach_gold(f,gold):
    g=gold.reset_index() if gold.index.name=='case_id' else gold
    z=f.merge(g[['case_id','reference_mean','reference_max']],on='case_id',validate='many_to_one');z['reference']=np.where(z.statistic=='mean',z.reference_mean,z.reference_max);z['error']=z.prediction-z.reference;return z
def load_bundle():return dict(models=pickle.loads((O/'models.pkl').read_bytes()),decisions=json.loads((O/'decisions.json').read_text()),base=json.loads((P/'v3/results/models.json').read_text()),regs=json.loads((P/'v4/results/regressions.json').read_text()),routes=json.loads((P/'v4/results/routes.json').read_text()))
def fetch_and_prepare():
    raw=D/'raw';raw.mkdir(exist_ok=True);manifest=[];lines=[]
    for station in STATIONS:
        path=raw/(station+'.txt');url='https://www.ndbc.noaa.gov/data/realtime2/'+station+'.txt'
        if path.exists(): b=path.read_bytes(); prior=json.loads((D/'source_manifest.json').read_text()) if (D/'source_manifest.json').exists() else [];meta=next((a for a in prior if a['station']==station),{})
        else:
            b=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'OceanCourseReproducibleResearch/1.0'}),timeout=45).read();assert b.startswith(b'#');path.write_bytes(b);meta={'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        manifest.append(dict(station=station,url=url,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),retrieved_utc=meta.get('retrieved_utc'),quality='automated realtime QC, not final archive'))
        text=b.decode('ascii');head=text.splitlines()[0].lstrip('#').split(); wi=head.index('WTMP')
        for ln,line in enumerate(text.splitlines(),1):
            if line.startswith('#') or not line.strip(): continue
            a=line.split()
            try:
                tt=pd.Timestamp(year=int(a[0]),month=int(a[1]),day=int(a[2]),hour=int(a[3]),minute=int(a[4]),tz='UTC');temp=float(a[wi]);assert -5<=temp<=45
                lines.append(dict(station=station,time=tt,hour=tt.floor('h'),value=temp,record_id=f'v5:{station}:{ln}',line=ln,source_sha256=manifest[-1]['sha256']))
            except (ValueError,AssertionError):continue
    dump(D/'source_manifest.json',manifest)
    f=pd.DataFrame(lines).sort_values('time').drop_duplicates(['station','hour'],keep='last');f.to_csv(D/'hourly_lineage.csv.gz',index=False)
    y=f.pivot(index='hour',columns='station',values='value');ix=pd.date_range(y.index.min().floor('D'),y.index.max().ceil('D'),freq='h',tz='UTC');y=y.reindex(ix);ids=f.pivot(index='hour',columns='station',values='record_id').reindex(ix);y.to_csv(D/'hourly.csv.gz')
    return y,ids
def build_snapshot(y,ids):
    cases=[];gold=[];index={t:i for i,t in enumerate(y.index)};arr={s:y[s].to_numpy() for s in y}
    dates=pd.date_range(y.index.min().ceil('D')+pd.Timedelta(days=3),y.index.max().floor('D')-pd.Timedelta(days=1),freq='D',tz='UTC')
    for s in STATIONS:
        for day in dates:
            ts=[day+pd.Timedelta(hours=h) for h in [0,6,12,18]];ix=[index[t] for t in ts];ref=arr[s][ix]
            if not np.isfinite(ref).all():continue
            order=sorted(range(4),key=lambda i:hashlib.sha256(f'{s}|{day}|{i}|20261113'.encode()).hexdigest())
            for scenario in SCENARIOS:
                n=int(scenario[-1]) if scenario.startswith('mask') else 4; hidden=order[:n];boundary=index[day]-(48 if scenario=='block72' else 0);lo=max(0,boundary-72);a=np.flatnonzero(np.isfinite(arr[s][lo:boundary]));ai=lo+int(a[-1]) if len(a) else None
                donors={};anchors={};sources={}
                for j in STATIONS:
                    if j==s:continue
                    donors[j]=[float(v) if np.isfinite(v) else None for v in arr[j][ix]];di=None
                    if ai is not None:
                        dl=max(0,ai-6);q=np.flatnonzero(np.isfinite(arr[j][dl:ai+1]));di=dl+int(q[-1]) if len(q) else None
                    anchors[j]=float(arr[j][di]) if di is not None else None;sources[j]=dict(current=[str(ids.loc[t,j]) if pd.notna(ids.loc[t,j]) else None for t in ts],anchor=str(ids.iloc[di][j]) if di is not None else None)
                    if scenario=='donor_missing' and j==sorted(a for a in STATIONS if a!=s)[0]:donors[j]=[None]*4;anchors[j]=None
                    if scenario in ['bias1','bias2']:donors[j]=[v+int(scenario[-1]) if v is not None else None for v in donors[j]]
                    if scenario=='relation_shift':donors[j]=[50-v if v is not None else None for v in donors[j]]
                cid=hashlib.sha256(f'v5|{s}|{day}|{scenario}'.encode()).hexdigest()[:24]
                c=dict(case_id=cid,station=s,date=str(day.date()),scenario=scenario,hours=[str(t) for t in ts],target_visible=[None if i in hidden else float(ref[i]) for i in range(4)],last_visible_value=float(arr[s][ai]) if ai is not None else None,last_visible_time=str(y.index[ai]) if ai is not None else None,donors=donors,donor_anchor=anchors,donor_sources=sources,target_anchor_source=str(ids.iloc[ai][s]) if ai is not None else None,target_visible_sources=[None if i in hidden else str(ids.loc[ts[i],s]) for i in range(4)],query_cutoff=str(day+pd.Timedelta(hours=19)))
                # Scenario is descriptive metadata; predictor never branches on it.
                cases.append(c);gold.append(dict(case_id=cid,station=s,date=str(day.date()),scenario=scenario,reference_mean=float(ref.mean()),reference_max=float(ref.max()),reference_values=ref.tolist(),reference_sources=[str(ids.loc[t,s]) for t in ts]))
    return cases,pd.DataFrame(gold)
def evaluate_phase():
    frozen=json.loads((O/'frozen.json').read_text());assert sha(__file__)==frozen['code_sha256'];assert sha(O/'models.pkl')==frozen['model_sha256']
    y,ids=fetch_and_prepare();cases,gold=build_snapshot(y,ids);assert len(cases)>0,'No complete cases in actual snapshot';dump(D/'public.json',cases);gold.to_json(D/'gold.json',orient='records',force_ascii=False,indent=2)
    bundle=load_bundle();records=[]
    for c in cases:
        st=old.state(c,bundle['base']);x,od=features(c,st);records.append((c,st,x,od))
    f=forecast(records,bundle);f.to_csv(O/'forecasts_without_gold.csv.gz',index=False)
    z=attach_gold(f,gold);cal=json.loads((O/'calibration.json').read_text())
    z['n_cal']=0;z['lower']=np.nan;z['upper']=np.nan;z['fixed_lower']=np.nan;z['fixed_upper']=np.nan
    for (context,method),ix in z.groupby(['context','method']).groups.items():
        q=cal[context+'|'+method];z.loc[ix,'n_cal']=q['n']
        if q['q'] is not None:
            z.loc[ix,'lower']=z.loc[ix,'lower_raw']-q['q'];z.loc[ix,'upper']=z.loc[ix,'upper_raw']+q['q'];z.loc[ix,'fixed_lower']=z.loc[ix,'prediction']-q['q_fixed'];z.loc[ix,'fixed_upper']=z.loc[ix,'prediction']+q['q_fixed']
    # v4 interval is frozen original, not refitted to new snapshot.
    oldp=pd.read_csv(P/'v4/results/predictions.csv.gz',dtype={'station':str});oldp=oldp[(oldp.split=='cal')&(oldp.method=='M_router')].drop_duplicates(['station','mask_count','age_bucket','statistic'])
    for _,r in oldp.iterrows():
        context=f'{r.station}|{int(r.mask_count)}|{r.age_bucket}|{r.statistic}'; ix=z.index[(z.context==context)&(z.method=='v4')];z.loc[ix,'lower']=z.loc[ix,'prediction']-r.q;z.loc[ix,'upper']=z.loc[ix,'prediction']+r.q
    z['width']=z.upper-z.lower;z['covered']=(z.reference>=z.lower)&(z.reference<=z.upper);z.to_csv(O/'predictions.csv.gz',index=False)
    metrics=[]
    for key,g in z.groupby(['scenario','statistic','method']):
        valid=g[g.lower.notna()];iscore=valid.width+20*np.maximum(valid.lower-valid.reference,0)+20*np.maximum(valid.reference-valid.upper,0)
        metrics.append(dict(scenario=key[0],statistic=key[1],method=key[2],n=len(g),RMSE=rmse(g.error),MAE=float(abs(g.error).mean()),n_interval=len(valid),PICP=float(valid.covered.mean()) if len(valid) else None,MPIW=float(valid.width.mean()) if len(valid) else None,interval_score=float(iscore.mean()) if len(valid) else None,fallback_rate=float(g.fallback.mean()),within_05=float((abs(g.error)<=.5).mean())))
    pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False)
    # Same v5 point estimates, alternate fixed intervals: isolate calibration design.
    zi=z[z.method=='v5'].copy();zi['fixed_covered']=(zi.reference>=zi.fixed_lower)&(zi.reference<=zi.fixed_upper);zi['fixed_width']=zi.fixed_upper-zi.fixed_lower
    zi.groupby(['scenario','statistic']).agg(n=('case_id','size'),CQR_PICP=('covered','mean'),CQR_MPIW=('width','mean'),fixed_PICP=('fixed_covered','mean'),fixed_MPIW=('fixed_width','mean')).reset_index().to_csv(O/'interval_ablation.csv',index=False)
    bootstrap(z);acquire(cases,gold);closure(z,cases,gold,frozen)
    summary={'dates':[str(y.index.min()),str(y.index.max())],'normal_requests':int((gold.scenario=='mask4').sum()),'total_requests':len(cases),'normal_by_station':gold[gold.scenario=='mask4'].groupby('station').size().to_dict(),'freeze_created_utc':frozen['created_utc'],'new_snapshot':'first downloaded after freeze; single pass, no post-result tuning','source_manifest_sha256':sha(D/'source_manifest.json')}
    dump(O/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True);print(pd.DataFrame(metrics).query("scenario in ['mask1','mask2','mask4','block72','relation_shift'] and statistic=='mean'").to_string(index=False),flush=True)
def bootstrap(z):
    rng=np.random.default_rng(20261113);rows=[]
    for scenario in ['mask1','mask2','mask4','block72','bias2','relation_shift']:
        p=z[(z.scenario==scenario)&(z.statistic=='mean')].pivot(index=['date','station'],columns='method',values='error').reset_index();dates=pd.date_range(p.date.min(),p.date.max());p['date']=pd.to_datetime(p.date);cols=['local','v4','v5_no_guard','v5'];ss=p.groupby('date')[cols].agg(lambda x:float((x*x).sum())).reindex(dates,fill_value=0).to_numpy();nn=p.groupby('date').size().reindex(dates,fill_value=0).to_numpy()
        for block in [3,7,14]:
            draw=[]
            for _ in range(2000):
                starts=rng.integers(0,max(1,len(dates)-block+1),size=math.ceil(len(dates)/block));ix=np.concatenate([np.arange(i,min(i+block,len(dates))) for i in starts])[:len(dates)];n=nn[ix].sum()
                if n:draw.append(np.sqrt(ss[ix].sum(axis=0)/n))
            draw=np.array(draw)
            for j in [0,1,2]:
                delta=draw[:,3]-draw[:,j];rows.append(dict(scenario=scenario,block_days=block,reference=cols[j],n=len(p),delta_RMSE=rmse(p.v5)-rmse(p[cols[j]]),lower=float(np.quantile(delta,.025)),upper=float(np.quantile(delta,.975))))
    pd.DataFrame(rows).to_csv(O/'bootstrap.csv',index=False)
def acquire(cases,gold):
    gd=gold.set_index('case_id');rows=[]
    for c in cases:
        if c['scenario']!='mask4':continue
        reference=gd.loc[c['case_id'],'reference_values'];day=pd.Timestamp(c['date'],tz='UTC');anchor=c['last_visible_value'];at=(pd.Timestamp(c['last_visible_time'])-day).total_seconds()/3600 if anchor is not None else -24.
        for policy in ['bridge','hashed','earliest']:
            known={};unseen=list(range(4));order=[]
            for budget in range(5):
                xs=([at] if anchor is not None else [])+[i*6 for i in known];vs=([anchor] if anchor is not None else [])+[known[i] for i in known]
                sort=np.argsort(xs);pred=np.interp([0,6,12,18],np.array(xs)[sort],np.array(vs)[sort]) if xs else np.repeat(np.nan,4)
                rows.append(dict(case_id=c['case_id'],station=c['station'],date=c['date'],policy=policy,budget=budget,prediction=float(np.mean(pred)),reference=float(np.mean(reference)),error=float(np.mean(pred)-np.mean(reference)),kind='direct' if budget==4 else 'estimate',revealed_hours=','.join(str(i*6) for i in order)))
                if not unseen:break
                if policy=='bridge': chosen=max(unseen,key=lambda i:(min(abs(i*6-t) for t in xs) if xs else 999,-i))
                elif policy=='earliest':chosen=min(unseen)
                else:chosen=min(unseen,key=lambda i:hashlib.sha256(f"{c['case_id']}|{i}".encode()).hexdigest())
                known[chosen]=reference[chosen];order.append(chosen);unseen.remove(chosen)
    f=pd.DataFrame(rows);f.to_csv(O/'acquisition_events.csv',index=False);f.groupby(['policy','budget']).agg(n=('case_id','size'),RMSE=('error',lambda a:rmse(a)),MAE=('error',lambda a:float(abs(a).mean()))).reset_index().to_csv(O/'acquisition_metrics.csv',index=False)
def closure(z,cases,gold,frozen):
    gd=gold.set_index('case_id');cd={c['case_id']:c for c in cases};claims=[];reveals=[];responses=[]
    for r in z[z.method=='v5'].itertuples():
        c=cd[r.case_id];valid=pd.notna(r.lower);claim_id=r.case_id+'|'+r.statistic;claims.append(dict(claim_id=claim_id,case_id=r.case_id,statistic=r.statistic,value=r.prediction,kind='estimate' if valid else 'hypothesis',lower=r.lower if valid else None,upper=r.upper if valid else None,model_sha256=frozen['model_sha256'],evidence_sources=c['target_visible_sources'],anchor_source=c['target_anchor_source']))
        gg=gd.loc[r.case_id];reveals.append(dict(claim_id=claim_id,reference=float(gg['reference_'+r.statistic]),reference_sources=gg.reference_sources,absolute_error=abs(r.error),covered=bool(r.covered),action='append audit only; original claim unchanged'))
        responses.append(dict(case_id=r.case_id,statistic=r.statistic,kind='estimate' if valid else 'hypothesis',text=f"{r.station}站{r.date}指定四个UTC小时代表值的{'均值' if r.statistic=='mean' else '最大值'}估计为{r.prediction:.2f}℃，区间[{r.lower:.2f}, {r.upper:.2f}]℃。此为估计，不是完整实测统计。" if valid else '当前历史校准支持不足，不能交付合格数值估计。',fallback=bool(r.fallback),next_action='补充该目标站缺失小时的WTMP记录，并核对单位、UTC时间及源行。'))
    for name,content in [('claims.jsonl',claims),('delayed_verification.jsonl',reveals),('responses.jsonl',responses)]: (O/name).write_text('\n'.join(json.dumps(a,ensure_ascii=False,allow_nan=False) for a in content),encoding='utf-8')
    original=sha(O/'claims.jsonl');probes=[]
    for version,kind,tau in [('v5','estimate',.5),('v4','estimate',.5),('v5','direct',.5),('v5','estimate',1.)]: probes.append(hashlib.sha256(f'{version}|{kind}|{tau}|{frozen["model_sha256"]}|{frozen["calibration_sha256"]}'.encode()).hexdigest())
    checks={'code_matches_frozen':sha(__file__)==frozen['code_sha256'],'models_match_frozen':sha(O/'models.pkl')==frozen['model_sha256'],'decisions_match_frozen':sha(O/'decisions.json')==frozen['decisions_sha256'],'calibration_matches_frozen':sha(O/'calibration.json')==frozen['calibration_sha256'],'download_after_freeze':all(a['retrieved_utc']>frozen['created_utc'] for a in json.loads((D/'source_manifest.json').read_text())),'all_scenarios_reported':len(set(z.scenario))==9,'cal_support_at_least100':bool(z.loc[z.lower.notna(),'n_cal'].ge(100).all()),'public_hidden_values_absent':all(all(v is None for v,source in zip(c['target_visible'],c['target_visible_sources']) if source is None) for c in cases),'anchor_strictly_before_target':all(pd.Timestamp(c['last_visible_time'])<pd.Timestamp(c['hours'][0]) for c in cases if c['last_visible_time']),'claims_preserved_after_audit':sha(O/'claims.jsonl')==original,'verification_one_to_one':len(claims)==len(reveals)==len(z[z.method=='v5']),'namespace_probes_distinct':len(set(probes))==4,'acquisition_full_restore_exact':bool(pd.read_csv(O/'acquisition_events.csv').query('budget==4').error.abs().le(1e-10).all())}
    dump(O/'audit.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),claim_count=len(claims),namespace_probes=probes));assert all(checks.values()),checks
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fit','evaluate'],required=True);args=ap.parse_args()
    if args.phase=='fit':fit_phase()
    else:evaluate_phase()
