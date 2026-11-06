import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from contiguous_fuxian.sparse_qwen_reprefill import _layer_causal_mask
from contiguous_fuxian.flexgen_pcache import _ordered_chunk_counts
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/flexgen'))
import my_pcache_fast as fast

class Round3Tests(unittest.TestCase):
    def test_masks_are_bit_exact(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for dtype in [torch.float16,torch.bfloat16,torch.float32]:
                for q in [0,1,7,33]:
                    for p in [0,1,16,65]:
                        expected=torch.full((1,1,q,p+q),torch.finfo(dtype).min,dtype=dtype,device=device)
                        expected[:,:,:,:p]=0
                        for row in range(q):expected[:,:,row,p:p+row+1]=0
                        actual=_layer_causal_mask(query_tokens=q,past_tokens=p,dtype=dtype,device=device)
                        self.assertTrue(torch.equal(actual,expected),(device,dtype,q,p))

    def test_metadata_order_and_counts_randomized(self):
        rng=np.random.default_rng(42)
        for n in [1,17,1027]:
            for chunk_size in [1,16,64]:
                for _ in range(30):
                    selected=rng.integers(0,n,size=n*2).tolist()
                    for reorder in [None,rng.permutation(n).tolist()]:
                        expected={}
                        for token in sorted(set(selected)):
                            physical=token if reorder is None else reorder[token]
                            chunk=physical//chunk_size
                            expected[chunk]=expected.get(chunk,0)+1
                        actual=_ordered_chunk_counts(selected,n,chunk_size,reorder)
                        self.assertEqual(list(actual.items()),list(expected.items()))
        self.assertEqual(_ordered_chunk_counts([],1,16),{})
        for ids in [[-1],[10]]:
            with self.assertRaises(ValueError):_ordered_chunk_counts(ids,10,16)

    def test_pool_waits_for_completion(self):
        class Event:
            ready=False
            waited=False
            def query(self):return self.ready
            def synchronize(self):self.ready=True;self.waited=True
        pool=fast.PinnedStagingPool(slab_bytes=1024,slots=1)
        slot,a=pool.acquire(torch.float16)
        event=Event();pool.release(slot,event)
        other,b=pool.acquire(torch.float16)
        self.assertTrue(event.waited)
        self.assertEqual(a.data_ptr(),b.data_ptr())
        self.assertEqual(pool.metrics()['allocated_peak_bytes'],1024)
        pool.release(other,None)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA test')
    def test_gpu_source_can_be_evicted_before_copy_completes(self):
        obj=fast.PrefixKVLayer.__new__(fast.PrefixKVLayer)
        obj.chunk_size=16;obj.shape=(2,16,2,4);obj.torch_dtype=torch.float16
        obj.disk_type='Chunk'
        expected=torch.arange(128,dtype=torch.float16).view(16,2,4)
        obj.tokens=[SimpleNamespace(token=SimpleNamespace(device='cuda:0',key=expected.cuda()))]
        stream=torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10000000)
            target=torch.empty((16,2,4),dtype=torch.float16,device='cuda:0')
            obj._batch_copy_chunk_sources(target,[0],[16],True)
        obj.tokens.clear()  # Simulate cache eviction on the main thread.
        churn=[torch.full((16,2,4),-7,dtype=torch.float16,device='cuda:0') for _ in range(100)]
        stream.synchronize()
        self.assertTrue(torch.equal(target.cpu(),expected))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA test')
    def test_copy_mixed_sources_tail_and_order(self):
        old_pool=fast._STAGING_POOL
        fast._STAGING_POOL=fast.PinnedStagingPool(slab_bytes=512,slots=2)
        self.addCleanup(setattr,fast,"_STAGING_POOL",old_pool)
        # Tiny slabs force multiple asynchronous flushes within a single KV load.
        # Load only the baseline method, leaving the module imports identical.
        import ast
        tree=ast.parse((ROOT/'audit/round3_baseline/vendor/flexgen/my_pcache_fast.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PrefixKVLayer')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prefetch_chunks_direct_pinned')
        ns={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),'baseline','exec'),ns)
        baseline=ns[method.name]
        with tempfile.TemporaryDirectory(dir=ROOT/'audit') as td:
            for split in [False,True]:
                obj=fast.PrefixKVLayer.__new__(fast.PrefixKVLayer)
                obj.chunk_size=16;obj.shape=(2,83,2,4);obj.torch_dtype=torch.float16
                obj.disk_type='KV_Division' if split else 'Chunk'
                keys=torch.arange(83*8,dtype=torch.float16).view(83,2,4)
                vals=keys+1000
                obj.tokens=[];obj.key_tokens=[];obj.value_tokens=[]
                for i,device in enumerate(['cuda:0','cpu','disk','cuda:0','cpu','disk']):
                    k=keys[i*16:(i+1)*16].clone();v=vals[i*16:(i+1)*16].clone()
                    def token(k,v,suffix):
                        both=k is not None and v is not None
                        shape=tuple(torch.stack([k,v]).shape if both else (k if k is not None else v).shape)
                        path=str(Path(td)/f'{split}_{i}_{suffix}')
                        if device=='disk':
                            data=torch.stack([k,v]).numpy() if both else (k if k is not None else v).numpy()
                            data.tofile(path)
                        return SimpleNamespace(device=device,key=k.to(device) if k is not None and device!='disk' else None,
                            value=v.to(device) if v is not None and device!='disk' else None,path=path,
                            dtype=np.float16,shape=shape,key_exist=k is not None,value_exist=v is not None)
                    if split:
                        obj.key_tokens.append(SimpleNamespace(token=token(k,None,'k')))
                        obj.value_tokens.append(SimpleNamespace(token=token(None,v,'v')))
                    else:obj.tokens.append(SimpleNamespace(token=token(k,v,'kv')))
                for ids in [[5,0,1,3,2,4],[2,0,1],[1,2],[0],[]]:
                    for need_k,need_v in [(True,True),(True,False),(False,True)]:
                        tid=torch.tensor(ids,dtype=torch.long)
                        ref=baseline(obj,tid,83,need_k,need_v)
                        got=obj._prefetch_chunks_direct_pinned(tid,83,need_k,need_v)
                        torch.cuda.synchronize()
                        for expected,actual in zip(ref[:3],got[:3]):
                            if expected is None:self.assertIsNone(actual)
                            else:self.assertTrue(torch.equal(expected,actual),(split,ids,need_k,need_v))
                # Reuse slabs aggressively without waiting for the output consumer.
                outputs=[]
                stream=torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    for _ in range(40):
                        outputs.append(obj._prefetch_chunks_direct_pinned(torch.tensor([1,2]),83,True,True))
                stream.synchronize()
                for k,v,ids,_ in outputs:
                    self.assertTrue(torch.equal(k.cpu(),keys[16:48]))
                    self.assertTrue(torch.equal(v.cpu(),vals[16:48]))
                self.assertLessEqual(fast._STAGING_POOL.metrics()['allocated_peak_bytes'],8*1024*1024)

if __name__=='__main__':unittest.main()
