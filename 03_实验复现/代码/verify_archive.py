from pathlib import Path
import sys,json,gzip,math,hashlib
R=Path(__file__).parent;P=R.parent;sys.path[:0]=[str(P/'.deps'),str(P/'v5')]
import run_experiment as e
import numpy as np,pandas as pd
O=R/'results';D=R/'data';z=pd.read_csv(O/'predictions.csv.gz',dtype={'station':str},low_memory=False);gold=pd.read_json(D/'gold.json',dtype={'station':str});cases=json.loads((D/'public.json').read_text());frozen=json.loads((O/'frozen.json').read_text());bundle=e.load_bundle()
before=e.sha(O/'claims.jsonl');records=[]
for c in cases:
 st=e.old.state(c,bundle['base']);x,od=e.features(c,st);records.append((c,st,x,od))
re=e.forecast(records,bundle).sort_values(['case_id','statistic','method']).reset_index(drop=True);saved=pd.read_csv(O/'forecasts_without_gold.csv.gz').sort_values(['case_id','statistic','method']).reset_index(drop=True);diff=float(np.max(abs(re.prediction-saved.prediction)))
raw={s:gzip.decompress((D/'raw'/f'{s}h2021.txt.gz').read_bytes()).decode('ascii').splitlines() for s in e.STATIONS};resolved=0;exact=True;direct=[]
for row in gold.itertuples():
 vals=[]
 for source in row.reference_sources:
  _,s,line=source.split(':');head=raw[s][0].lstrip('#').split();a=raw[s][int(line)-1].split();vals.append(float(a[head.index('WTMP')]));resolved+=1
 exact &= math.isclose(sum(vals)/4,row.reference_mean,abs_tol=1e-12) and math.isclose(max(vals),row.reference_max,abs_tol=1e-12)
 for stat,value in [('mean',sum(vals)/4),('max',max(vals))]:direct.append(dict(claim_id=row.case_id+'|'+stat+'|restored',parent_claim_id=row.case_id+'|'+stat,kind='direct',value=value,reference_sources=row.reference_sources,operation='new claim after all four source observations restored; old estimate preserved'))
(O/'restored_direct_claims.jsonl').write_text('\n'.join(json.dumps(c,ensure_ascii=False) for c in direct),encoding='utf-8')
# Paired evidence-budget comparisons for the previously unseen 2021 year.
ac=pd.read_csv(O/'acquisition_events.csv');rng=np.random.default_rng(20261113);br=[]
for budget in [1,2,3]:
 p=ac[ac.budget==budget].pivot(index=['date','station'],columns='policy',values='error').reset_index();p['date']=pd.to_datetime(p.date);dates=pd.date_range(p.date.min(),p.date.max());cols=['bridge','hashed','earliest'];ss=p.groupby('date')[cols].agg(lambda a:float((a*a).sum())).reindex(dates,fill_value=0).to_numpy();nn=p.groupby('date').size().reindex(dates,fill_value=0).to_numpy();draws=[]
 for _ in range(2000):
  starts=rng.integers(0,len(dates)-7+1,size=math.ceil(len(dates)/7));ix=np.concatenate([np.arange(i,i+7) for i in starts])[:len(dates)];draws.append(np.sqrt(ss[ix].sum(0)/nn[ix].sum()))
 draws=np.array(draws)
 for j in [1,2]:
  de=draws[:,0]-draws[:,j];br.append(dict(budget=budget,reference=cols[j],delta_RMSE=e.rmse(p.bridge)-e.rmse(p[cols[j]]),lower=float(np.quantile(de,.025)),upper=float(np.quantile(de,.975)),block_days=7,n=len(p)))
pd.DataFrame(br).to_csv(O/'acquisition_bootstrap.csv',index=False)
checks={'public_forecasts_reproduce_1e12':diff<1e-12,'every_hidden_reference_resolves_to_source':bool(exact),'original_claim_log_unchanged':e.sha(O/'claims.jsonl')==before,'restored_claims_one_to_one':len(direct)==len(gold)*2,'frozen_validation_code_unchanged':e.sha(R/'validate.py')==frozen['code_sha256'],'nine_scenarios_retained':z.scenario.nunique()==9,'all_methods_same922_cases':bool(z.groupby(['scenario','statistic','method']).size().eq(922).all())}
e.dump(O/'independent_audit.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),source_line_resolutions=resolved,max_prediction_difference=diff,restored_direct_claim_count=len(direct)));assert all(checks.values());print(json.dumps(checks));print(pd.DataFrame(br).to_string(index=False))
