from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util,sys
import numpy as np
import pytest
import torch
from contiguous_fuxian.sharedkv_v23.cold import validate_persisted_resident
def test_register_existing_resident_without_writing(tmp_path):
 x=torch.arange(32,dtype=torch.float16).reshape(4,8);p=tmp_path/'kv.npy';p.write_bytes(x.numpy().tobytes())
 before=p.stat()
 t=NS(path=str(p),file_exist=False,key=x,value=None,key_exist=True,value_exist=False)
 assert validate_persisted_resident(t,tmp_path)
 assert t.file_exist and t.shape==(4,8) and t.dtype==np.dtype('float16')
 assert p.stat().st_mtime_ns==before.st_mtime_ns
 assert not validate_persisted_resident(t,tmp_path)
def test_registration_rejects_content_mismatch(tmp_path):
 p=tmp_path/'kv.npy';p.write_bytes(bytes(8));t=NS(path=str(p),file_exist=False,key=torch.ones(4,dtype=torch.float16),value=None,key_exist=True,value_exist=False)
 with pytest.raises(RuntimeError,match='content mismatch'):validate_persisted_resident(t,tmp_path)
 assert not t.file_exist and p.read_bytes()==bytes(8)
@pytest.mark.parametrize('fail',[False,True])
def test_launcher_restores_global_patches(tmp_path,monkeypatch,fail):
 root=Path(__file__).resolve().parents[1]
 spec=importlib.util.spec_from_file_location('launcher_v23',root/'scripts/run_sharedkv_v23.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from contiguous_fuxian.sharedkv_v21.runtime import Runtime
 from contiguous_fuxian.flexgen_pcache import FlexGenPcacheStore
 import contiguous_fuxian.flexgen_qwen_reprefill as runner
 orig=(Runtime.install,FlexGenPcacheStore.new_layer_loader,runner.main)
 def fake_run(*a,**k):
  assert Runtime.install is not orig[0]
  assert FlexGenPcacheStore.new_layer_loader is not orig[1]
  assert runner.main is not orig[2]
  if fail:raise RuntimeError('test downstream failure')
 monkeypatch.setattr(m.runpy,'run_path',fake_run)
 for i in range(2):
  monkeypatch.setattr(sys,'argv',['run','--cold','ssd-cold','--owned-cache-root',str(tmp_path),'--pcache-root',str(tmp_path),'--cold-log',str(tmp_path/f'{i}.jsonl'),'--','--mode','identity','--','--task','trec','--warmup','1','--samples','2'])
  if fail:
   with pytest.raises(RuntimeError,match='downstream failure'):m.main()
  else:m.main()
  assert (Runtime.install,FlexGenPcacheStore.new_layer_loader,runner.main)==orig

@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_fast_directory_audit_rejects_aliases(tmp_path,kind):
 from contiguous_fuxian.sharedkv_v23.cold import audit_files
 import os
 p=tmp_path/'original';p.write_bytes(bytes(32));q=tmp_path/'alias'
 if kind=='symlink':q.symlink_to(p)
 else:os.link(p,q)
 with pytest.raises(ValueError):audit_files([q],tmp_path)
