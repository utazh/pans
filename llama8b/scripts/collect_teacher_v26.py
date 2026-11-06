
from pathlib import Path
import argparse,runpy,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser();p.add_argument('--teacher-records',required=True);p.add_argument('--split',choices=['fit','validation'],required=True);p.add_argument('rest',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.rest[1:] if a.rest[:1]==['--'] else a.rest
assert '--diagnostic' in argv and argv[argv.index('--warmup')+1]=='0'
from contiguous_fuxian.sharedkv_v25.runtime import Runtime
from contiguous_fuxian.sharedkv_v24.capture import ScoreCollector
old=Runtime.install
def install(self,runner):
 old(self,runner);ScoreCollector(a.teacher_records,a.split).install(self,runner)
root=Path(a.teacher_records);root.mkdir(parents=True,exist_ok=False)
(root/'CALIBRATION_ONLY.json').write_text(json.dumps(dict(split=a.split,formal_timing_eligible=False)))
Runtime.install=install
try:
 sys.argv=[str(ROOT/'scripts/run_evidence_v26.py'),*argv];runpy.run_path(sys.argv[0],run_name='__main__')
finally:Runtime.install=old
