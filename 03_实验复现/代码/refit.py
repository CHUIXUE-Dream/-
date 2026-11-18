"""Offline numerical reproduction into a fresh directory; never replaces frozen outputs."""
from pathlib import Path
import sys,datetime,json
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P/'.deps'),str(P/'v5')]
import run_experiment as e
import pandas as pd,numpy as np
run=P/'v6/reproduction_runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True,exist_ok=False);e.O=run
e.fit_phase();bundle=e.load_bundle();checks={}
for folder in ['v5','v6']:
 cases=json.loads((P/folder/'data/public.json').read_text(encoding='utf-8'));records=[]
 for c in cases:
  st=e.old.state(c,bundle['base']);x,od=e.features(c,st);records.append((c,st,x,od))
 forecast=e.forecast(records,bundle).sort_values(['case_id','statistic','method']).reset_index(drop=True)
 frozen=pd.read_csv(P/folder/'results/forecasts_without_gold.csv.gz').sort_values(['case_id','statistic','method']).reset_index(drop=True)
 diffs={key:float(np.max(abs(forecast[key]-frozen[key]))) for key in ['prediction','lower_raw','upper_raw']}
 checks[folder]={'rows':len(forecast),'max_absolute_differences':diffs,'passed':all(v<1e-12 for v in diffs.values())}
checks['calibration_numeric_identical']=json.loads((run/'calibration.json').read_text())==json.loads((P/'v5/results/calibration.json').read_text())
checks['fresh_output_dir']=str(run);e.dump(run/'reproduction_verification.json',checks);e.dump(P/'v6/results/full_reproduction_verification.json',checks)
assert all(checks[v]['passed'] for v in ['v5','v6']) and checks['calibration_numeric_identical'];print(json.dumps(checks,indent=2))
