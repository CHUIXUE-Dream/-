from pathlib import Path
import sys,json,ast,hashlib,math
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
src=(R/'run_workflow.py').read_text(encoding='utf-8');node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='stages');exec(ast.get_source_segment(src,node),globals())
cases=json.loads((R/'data/public.json').read_text());gold=pd.read_json(R/'data/gold.json',dtype={'station':str}).set_index('case_id');ev=pd.read_csv(R/'results/workflow_events.csv.gz',dtype={'station':str});base=pd.read_csv(R/'results/point_predictions.csv.gz');base=base[(base.method=='v4')&(base.statistic=='mean')].set_index('case_id');prefix=True;count=0
for c in cases[::max(1,len(cases)//30)]:
 for pol in ['H_adapt2','E_adapt2','T_adapt4']:
  r=ev[(ev.case_id==c['case_id'])&(ev.policy==pol)&(ev.tau==.25)].iloc[0];ref=list(gold.loc[c['case_id']].reference_values);seen={int(h)//6 for h in str(r.revealed_hours).split(',') if h!='nan'} if pd.notna(r.revealed_hours) else set()
  for i in range(4):
   if i not in seen:ref[i]+=99
  a=stages(c,ref,pol,float(base.loc[c['case_id']].prediction))[int(r.budget)];prefix &= math.isclose(a['prediction'],r.prediction,abs_tol=1e-12);count+=1
summary=pd.read_csv(R/'results/workflow_metrics.csv');same=True
for row in summary.itertuples():
 g=ev[(ev.policy==row.policy)&(ev.tau==row.tau)];same &= math.isclose(g.qualified.mean(),row.qualified_rate,abs_tol=1e-12) and math.isclose(g.budget.mean(),row.avg_records,abs_tol=1e-12)
checks={'unrevealed_future_values_do_not_change_current_prediction':prefix,'all24_metric_cells_recompute':same,'all_frozen_checks':all(json.loads((R/'results/audit.json').read_text())['checks'].values())}
out={'checks':checks,'prefix_probes':count,'passed':sum(checks.values()),'total':len(checks)};(R/'results/independent_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out));assert all(checks.values())
