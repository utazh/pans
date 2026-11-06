
import argparse,runpy,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
p=argparse.ArgumentParser();p.add_argument('--audit-output',required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.rest[1:] if a.rest[:1]==['--'] else a.rest
assert '--diagnostic' in argv
from contiguous_fuxian.sharedkv_v25.runtime import Runtime
from contiguous_fuxian.sharedkv_v24.restore_audit import install as audit_install
old=Runtime.install
def install(self,runner):
 old(self,runner);audit_install(runner,a.audit_output)
Runtime.install=install
try:
 sys.argv=[str(r/'scripts/run_native_prefix_v25.py'),*argv];runpy.run_path(sys.argv[0],run_name='__main__')
finally:Runtime.install=old
