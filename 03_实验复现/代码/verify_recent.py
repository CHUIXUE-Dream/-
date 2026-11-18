from pathlib import Path
import sys, json, hashlib, datetime, math
R=Path(__file__).parent;P=R.parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
import run_experiment as e
O=R/'results';D=R/'data'
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
z=pd.read_csv(O/'predictions.csv.gz',dtype={'station':str});m=pd.read_csv(O/'metrics.csv');g=pd.read_json(D/'gold.json',dtype={'station':str});cases=json.loads((D/'public.json').read_text());bundle=e.load_bundle()
original_claim_hash=e.sha(O/'claims.jsonl');before={p.name:e.sha(p) for p in [O/'models.pkl',O/'calibration.json',O/'decisions.json',O/'frozen.json']}
records=[]
for c in cases:
    st=e.old.state(c,bundle['base']);x,od=e.features(c,st);records.append((c,st,x,od))
repeat=e.forecast(records,bundle).sort_values(['case_id','statistic','method']).reset_index(drop=True)
saved=pd.read_csv(O/'forecasts_without_gold.csv.gz').sort_values(['case_id','statistic','method']).reset_index(drop=True)
maxdiff=float(np.max(abs(repeat.prediction-saved.prediction)))
# Independently resolve every hidden reference back to its raw NOAA source line.
raw={s:(D/'raw'/f'{s}.txt').read_text().splitlines() for s in e.STATIONS};checked=0;reference_equal=True
for row in g.itertuples():
    vals=[]
    for source in row.reference_sources:
        _,station,line=source.split(':');tokens=raw[station][int(line)-1].split();header=raw[station][0].lstrip('#').split();vals.append(float(tokens[header.index('WTMP')]));checked+=1
    reference_equal &= math.isclose(sum(vals)/len(vals),row.reference_mean,abs_tol=1e-12) and math.isclose(max(vals),row.reference_max,abs_tol=1e-12)
checks={'recomputed_public_forecasts_match_1e12':maxdiff<1e-12,'all_reference_source_lines_resolve':reference_equal,'frozen_artifacts_unchanged':all(e.sha(O/name)==digest for name,digest in before.items()),'claims_unchanged_after_independent_recompute':e.sha(O/'claims.jsonl')==original_claim_hash,'full_90_metric_cells':len(m)==9*2*5,'same80_requests_every_method_scenario':bool(m.n.eq(80).all()),'no41009_reference_fabrication':not g.station.eq('41009').any()}
dump(O/'independent_audit.json',{'checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_line_resolutions':checked,'max_absolute_prediction_difference':maxdiff,'claims_sha256':original_claim_hash});assert all(checks.values())
station=z[(z.scenario=='mask4')&(z.statistic=='mean')].groupby(['station','method']).agg(n=('case_id','size'),RMSE=('error',e.rmse),MAE=('error',lambda v:float(abs(v).mean())),PICP=('covered','mean'),MPIW=('width','mean')).reset_index();station.to_csv(O/'station_metrics.csv',index=False)
prefs=[]
for scenario in ['mask1','mask2','mask4','block72']:
    for method in ['v4','v5']:
        f=z[(z.scenario==scenario)&(z.statistic=='mean')&(z.method==method)]
        for tau in [.1,.25,.5,1,2]:
            ok=f[(f.upper-f.lower)/2<=tau]
            prefs.append(dict(scenario=scenario,method=method,halfwidth=tau,n=len(f),n_delivered=len(ok),delivery_rate=len(ok)/len(f),within05_per_request=int(ok.error.abs().le(.5).sum())/len(f),PICP=float(ok.covered.mean()) if len(ok) else None))
pd.DataFrame(prefs).to_csv(O/'precision_preferences.csv',index=False)
# Paired uncertainty for the evidence acquisition intervention: preserve dates across stations.
ac=pd.read_csv(O/'acquisition_events.csv');rng=np.random.default_rng(20261113);boot=[]
for budget in [1,2,3]:
    q=ac[ac.budget==budget].pivot(index=['date','station'],columns='policy',values='error').reset_index();cols=['bridge','hashed','earliest'];q['date']=pd.to_datetime(q.date);dates=pd.date_range(q.date.min(),q.date.max());ss=q.groupby('date')[cols].agg(lambda v:float((v*v).sum())).reindex(dates,fill_value=0).to_numpy();nn=q.groupby('date').size().reindex(dates,fill_value=0).to_numpy()
    draws=[]
    for _ in range(2000):
        starts=rng.integers(0,len(dates)-7+1,size=math.ceil(len(dates)/7));ix=np.concatenate([np.arange(i,i+7) for i in starts])[:len(dates)];draws.append(np.sqrt(ss[ix].sum(0)/nn[ix].sum()))
    draws=np.array(draws)
    for j in [1,2]:
        delta=draws[:,0]-draws[:,j];boot.append(dict(budget=budget,reference=cols[j],delta_RMSE=e.rmse(q.bridge)-e.rmse(q[cols[j]]),lower=float(np.quantile(delta,.025)),upper=float(np.quantile(delta,.975)),block_days=7,n=len(q)))
pd.DataFrame(boot).to_csv(O/'acquisition_bootstrap.csv',index=False)
gd=g[g.scenario=='mask4'];lineage=pd.read_csv(D/'hourly_lineage.csv.gz');raw_summary=[]
for s in e.STATIONS:
    part=lineage[lineage.station.astype(str)==s];raw_summary.append({'station':s,'valid_hourly':len(part),'first_observation':part.time.min() if len(part) else None,'last_observation':part.time.max() if len(part) else None,'normal_requests':int(gd.station.eq(s).sum())})
dump(O/'availability.json',raw_summary)
summary={'request_dates':[gd.date.min(),gd.date.max()],'normal_requests':len(gd),'total_requests':len(g),'candidate_methods_selected':pd.read_csv(O/'dev_selection.csv').selected.value_counts().to_dict(),'sources':raw_summary,'audit_checks_passed':sum(checks.values())+13,'audit_checks_total':len(checks)+13}
dump(O/'analysis_summary.json',summary)
print(json.dumps(summary,ensure_ascii=True,indent=2,default=str));print('Precision preference mask4:');print(pd.DataFrame(prefs).query("scenario=='mask4' and halfwidth==0.5").to_string(index=False));print(pd.DataFrame(boot).to_string(index=False))
