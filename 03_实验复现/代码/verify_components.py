"""Independent report-level recomputation of E1/E4 summaries and E2/E3 invariants."""
from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
A=P/'report_work/thesis05/analysis';checks={}
e=pd.read_csv(P/'v3/results/evidence_predictions.csv')
m=pd.read_csv(P/'v3/results/evidence_metrics.csv');ok=True
for r in m.itertuples():
 g=e[(e.split==r.split)&(e.method==r.method)]
 vals={'n_targets':len(g),'unsupported_n':int((~g.gold_direct&g.delivered_direct).sum()),'unsupported_den':int((~g.gold_direct).sum()),'false_refusal_n':int((g.gold_direct&~g.delivered_direct).sum()),'false_refusal_den':int(g.gold_direct.sum()),'coverage':g.delivered_direct.mean(),'direct_accuracy':g[g.delivered_direct].numeric_correct.mean()}
 ok &= all(np.isclose(v,getattr(r,k),rtol=0,atol=1e-9) for k,v in vals.items())
checks['E1_all_splits_from_target_records']=bool(ok)
events=pd.read_csv(P/'v3/results/cache_events.csv.gz');saved=pd.read_csv(P/'v3/results/cache_metrics.csv')
ok=True
for r in saved.itertuples():
 g=events[(events.seed==r.seed)&(events.repeat==r.repeat)&(events.policy==r.policy)]
 vals={'n':len(g),'P50_ms':g.latency_ms.quantile(.5),'P95_ms':g.latency_ms.quantile(.95),'unsafe_rate':g.unsafe.mean()}
 ok &= all(np.isclose(v,getattr(r,k),rtol=0,atol=1e-9) for k,v in vals.items())
checks['E4_all120_runs_from120000_events']=bool(ok and len(events)==120000)
checks['E4_Kp_lowest_safe_P50']=saved.groupby('policy')[['P50_ms','unsafe_rate']].median().query('unsafe_rate==0').P50_ms.idxmin()=='A_no_hotness'
checks['E4_K0_lowest_safe_P95']=saved.groupby('policy')[['P95_ms','unsafe_rate']].median().query('unsafe_rate==0').P95_ms.idxmin()=='C0_resident_no_cache'
for ver in ['v5','v6']:
 for fname in ['audit.json','independent_audit.json']:
  audit=json.loads((P/ver/'results'/fname).read_text(encoding='utf-8'))
  checks[ver+'_'+fname]=all(audit['checks'].values())
checks['E2_E3_frozen_output_analysis']=all(json.loads((A/'analysis_audit.json').read_text(encoding='utf-8'))['checks'].values())
r=json.loads((P/'v6/results/full_reproduction_verification.json').read_text())
checks['independent_refit90180_rows']=r['v5']['passed'] and r['v6']['passed'] and r['v5']['rows']+r['v6']['rows']==90180 and r['calibration_numeric_identical']
rep={'checks':checks,'passed':sum(checks.values()),'total':len(checks),'scope':'Recomputed saved E1/E4 event summaries, audited E2/E3 frozen outputs, and retained independent training-reproduction evidence; no new test selection or timing measurement.'}
(P/'report_work/thesis05/experiment_verification.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rep,ensure_ascii=True,indent=2));assert all(checks.values())
