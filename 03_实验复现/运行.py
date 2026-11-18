"""Current submission entry. Materialize byte-frozen modules in a temporary workspace.
Only required shared modules are shipped once; logical legacy IDs preserve frozen
hashes and are mapped to functional submission folders. No external download.
"""
from pathlib import Path
import json,hashlib,shutil,sys,os,tempfile,subprocess,datetime,argparse
S=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser();parser.add_argument('--full-refit',action='store_true');parser.add_argument('--figures',action='store_true');a=parser.parse_args()
rows=json.loads((S/'03_实验复现/路径映射.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for r in rows:assert sha(S/r['path'])==r['sha256'],r['path']
T=Path(tempfile.mkdtemp(prefix='ocean_course_replay_'));O=S/'07_核验/本次复现';O.mkdir(parents=True,exist_ok=True)
for r in rows:
 p=T/r['logical_path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/r['path'],p)
(T/'report_work/thesis06').mkdir(exist_ok=True)
steps=['v5/analyze_and_verify.py','v6/final_audit.py','report_work/analyze05.py','verify_thesis05.py','v7/reproduce_frozen.py','v7/analyze_workflow.py','v7/independent_verify.py','v7/replay_execution.py']
if a.full_refit:steps.append('reproduce_from_training04.py')
steps.append('verify_thesis06.py')
if a.figures:steps.append('report_work/figures_current.py')
results=[]
env=dict(os.environ);env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
with (O/'运行日志.txt').open('w',encoding='utf-8') as log:
 for step in steps:
  print('Running '+step,flush=True)
  r=subprocess.run([sys.executable,str(T/step)],cwd=T,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  log.write(r.stdout.decode('utf-8',errors='replace')+'\n');log.flush();results.append(dict(module=step,exit_code=r.returncode));assert r.returncode==0,step+' failed; see log'
for key,file in [('汇总检查','report_work/thesis06/experiment_verification.json'),('组合冻结检查','v7/results/audit.json'),('组合独立检查','v7/results/independent_audit.json'),('实际执行检查','v7/results/execution_audit.json')]:shutil.copy2(T/file,O/(key+'.json'))
if a.full_refit:shutil.copy2(T/'v6/results/full_reproduction_verification.json',O/'训练复现检查.json')
if a.figures:
 F=T/'report_work/figures06';M=json.loads((F/'manifest.json').read_text(encoding='utf-8'));target=T/'figures_current';target.mkdir(exist_ok=True)
 figure_checks=[]
 for i,(name,item) in enumerate(M.items(),1):
  for ext in ['png','svg','pdf','csv']:
   q=F/(name+'.'+ext)
   if q.exists():shutil.copy2(q,target/(f'figure_{i:02d}.'+ext))
  original=S/'06_可视化'/f'figure_{i:02d}.csv';new=target/f'figure_{i:02d}.csv'
  if original.exists():
   import pandas as pd
   pd.testing.assert_frame_equal(pd.read_csv(original),pd.read_csv(new),check_exact=False,rtol=1e-10,atol=1e-12);figure_checks.append(dict(figure=i,numeric_csv_identical=True))
 (O/'绘图检查.json').write_text(json.dumps(dict(figures=len(M),data_figure_checks=figure_checks),ensure_ascii=False,indent=2),encoding='utf-8')
summary=dict(passed=True,input_files_verified=len(rows),steps=results,scope='Recomputed saved E1/E4 records, E2/E3 statistics and frozen E5, including actual acquisition execution. Original byte-frozen scientific modules are unchanged; only directory mapping is new.',full_refit=a.full_refit,figures=a.figures,temporary_workspace=str(T))
(O/'复现结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: current submission reproduction completed.')
