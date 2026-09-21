#!/usr/bin/env python3
"""v24 capture/audit flags -- v23 cold flags -- v21 runtime flags -- round4 flags.

Normal quality/cold runs can use the existing v23 entry with exported readers.
This wrapper adds only compact calibration capture or read-attempt diagnostics.
"""
from pathlib import Path
import argparse,json,runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def get(args,flag,default=None):
    if flag not in args:return default
    i=args.index(flag)
    if i+1==len(args):raise ValueError('Missing flag value: '+flag)
    return args[i+1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-records');p.add_argument('--split',choices=['fit','validation'])
    p.add_argument('--audit-restore');p.add_argument('rest',nargs=argparse.REMAINDER)
    a=p.parse_args();argv=a.rest
    if argv and argv[0]=='--':argv=argv[1:]
    if argv.count('--')!=2:raise ValueError('Pass v23 options -- v21 options -- round4 options')
    i=argv.index('--');v23=argv[:i];tail=argv[i+1:];j=tail.index('--');v21=tail[:j];round4=tail[j+1:]
    if bool(a.teacher_records)==bool(a.audit_restore):raise ValueError('Select exactly capture OR restore-audit; use old entry for formal runs')
    if '--diagnostic' not in v21:raise ValueError('Capture/audit require --diagnostic; not formal timing')
    if a.teacher_records:
        if not a.split or get(v21,'--mode')!='identity':raise ValueError('Teacher capture requires identity and explicit split')
        if int(get(round4,'--warmup','32'))!=0 or int(get(round4,'--max-tokens','1'))!=1:
            raise ValueError('Capture requires warmup=0, max-tokens=1; not a cold benchmark')
        if get(v23,'--cold','none')!='none' or any(x in v23+v21 for x in ('--capture-dir','--logits','--trace-dir','--replay-dir','--selection-plan')):
            raise ValueError('Compact clean capture must not overlap other capture/replay/cold modes')
        root=Path(a.teacher_records);root.mkdir(parents=True,exist_ok=False)
        (root/'CALIBRATION_ONLY.json').write_text(json.dumps({'split':a.split,'formal_timing_eligible':False}))
    if a.audit_restore and Path(a.audit_restore).exists():raise FileExistsError(a.audit_restore)
    from contiguous_fuxian.sharedkv_v24.guard import check
    check(ROOT)
    from contiguous_fuxian.sharedkv_v21.runtime import Runtime
    old_install=Runtime.install
    def install(self,runner):
        old_install(self,runner)
        if a.teacher_records:
            from contiguous_fuxian.sharedkv_v24.capture import ScoreCollector
            ScoreCollector(a.teacher_records,a.split).install(self,runner)
        else:
            from contiguous_fuxian.sharedkv_v24.restore_audit import install as audit_install
            audit_install(runner,a.audit_restore)
    Runtime.install=install
    sys.argv=[str(ROOT/'scripts/run_sharedkv_v23.py'),*argv]
    try:runpy.run_path(str(ROOT/'scripts/run_sharedkv_v23.py'),run_name='__main__')
    finally:Runtime.install=old_install

if __name__=='__main__':main()
