"""Aggregate original E1-E4 checks and new frozen E5 recomputation evidence."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
base=json.loads((P/'report_work/thesis05/experiment_verification.json').read_text())
checks=dict(base['checks'])
for name,file in [('E5_frozen','audit.json'),('E5_independent','independent_audit.json')]:
    a=json.loads((P/'v7/results'/file).read_text());checks[name]=all(a['checks'].values())
f=json.loads((P/'v7/results/frozen.json').read_text())
checks['E5_frozen'] &= hashlib.sha256((P/'v7/PROTOCOL.md').read_bytes()).hexdigest()==f['protocol_sha256'] and hashlib.sha256((P/'v5/results/models.pkl').read_bytes()).hexdigest()==f['base_model_sha256']
a=json.loads((P/'v7/results/execution_audit.json').read_text());checks['E5_execution']=a['passed'] and a['mismatches']==0
out=dict(checks=checks,passed=sum(checks.values()),total=len(checks),scope='E1-E4 saved-event recomputation and retained independent refit; E5 byte-frozen script replay, 24 metric cells, 93 prefix probes and actual cache-miss acquisition execution.')
(P/'report_work/thesis06/experiment_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2));assert all(checks.values())
