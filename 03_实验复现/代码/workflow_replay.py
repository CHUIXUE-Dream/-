"""Replay the byte-frozen E5 script with atomic source-manifest writes.

The original first-download script writes a growing manifest after each station.
On replay all raw files exist; intermediate partial writes would remove subsequent
stations before their original retrieval times are read. Suppress only incomplete
manifest writes. The frozen decision code, timestamps and numerical rules do not
change. Do not invoke run_workflow.py directly on an existing complete archive.
"""
from pathlib import Path
import json,runpy,hashlib
R=Path(__file__).resolve().parent;manifest=R/'data/source_manifest.json'
snapshot=manifest.read_bytes();expected=len(json.loads(snapshot));original_write=Path.write_text
frozen=json.loads((R/'results/frozen.json').read_text())
assert hashlib.sha256((R/'run_workflow.py').read_bytes()).hexdigest()==frozen['code_sha256']
def atomic_manifest_write(path,data,*args,**kwargs):
    if path.resolve()==manifest.resolve() and len(json.loads(data))<expected:
        return len(data)
    return original_write(path,data,*args,**kwargs)
try:
    Path.write_text=atomic_manifest_write
    runpy.run_path(str(R/'run_workflow.py'),run_name='__main__')
finally:
    Path.write_text=original_write
    manifest.write_bytes(snapshot)
assert hashlib.sha256((R/'run_workflow.py').read_bytes()).hexdigest()==frozen['code_sha256']
print('Frozen E5 replay passed; source manifest original bytes preserved.')
