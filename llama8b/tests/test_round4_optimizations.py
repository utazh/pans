from __future__ import annotations
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from contiguous_fuxian.flexgen_pcache import (
    FlexGenLayerLoader, FlexGenPcacheConfig, prefetch_source_tensor_tokens,
)
from contiguous_fuxian.sparse_qwen_reprefill import PrefixStoreInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/flexgen'))
import my_pcache_fast as fast

class Round4Tests(unittest.TestCase):
    def test_native_host_gather_preserves_bits_and_destination(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            parts=[torch.arange(n*16,dtype=dtype).reshape(n,2,8)[:,:,::2]
                   for n in [16,7,16,1]]
            expected=torch.cat(parts)
            dest=torch.empty_like(expected, pin_memory=torch.cuda.is_available())
            ptr=dest.data_ptr()
            if dtype==torch.bfloat16:
                inputs=parts
            else:
                inputs=[x.numpy() for x in parts]
                for x in inputs:x.flags.writeable=False
            fast._gather_host_parts(inputs,dest)
            self.assertEqual(ptr,dest.data_ptr())
            self.assertTrue(torch.equal(dest,expected))
            self.assertTrue(torch.equal(torch.cat(parts),expected))

    def test_source_counts_match_baseline_and_refresh_residency(self):
        tree=ast.parse((ROOT/'audit/round4_baseline/src/contiguous_fuxian/flexgen_pcache.py').read_text())
        node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prefetch_source_tensor_tokens')
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'baseline','exec'),ns)
        baseline=ns[node.name]
        rng=np.random.default_rng(73)
        for prefix in [1,17,1027]:
            chunk_size=16;count=(prefix+15)//16
            layer=SimpleNamespace(chunk_num=count,device_map=np.array(['cpu']*(count*2),dtype='<U16'))
            cache=SimpleNamespace(cache=[SimpleNamespace(layers=[layer])])
            for _ in range(50):
                layer.device_map[:]=rng.choice(['cpu','disk','cuda:0','cuda:1'],count*2)
                ids=rng.integers(0,prefix,size=prefix*2).tolist()
                args=dict(prefix_id=0,layer=0,token_ids=ids,chunk_size=chunk_size,prefix_tokens=prefix)
                self.assertEqual(prefetch_source_tensor_tokens(cache,**args),baseline(cache,**args))
            layer.device_map[:]='cpu'
            self.assertEqual(prefetch_source_tensor_tokens(cache,prefix_id=0,layer=0,token_ids=[],chunk_size=16,prefix_tokens=prefix),dict(gpu=0,cpu=0,disk=0))
            with self.assertRaises(ValueError):
                prefetch_source_tensor_tokens(cache,prefix_id=0,layer=0,token_ids=[-1],chunk_size=16,prefix_tokens=prefix)
            layer.device_map[0]='invalid'
            with self.assertRaises(RuntimeError):
                prefetch_source_tensor_tokens(cache,prefix_id=0,layer=0,token_ids=[0],chunk_size=16,prefix_tokens=prefix)

    def test_mapping_snapshot_and_reconfigured_selection(self):
        mapping=[[2,0,3,1,7,5,6,4]]
        inverse=[np.argsort(mapping[0]).tolist()]
        loader=FlexGenLayerLoader(pcache=object(),prefix_id=0,
            info=PrefixStoreInfo('rte',8,1,4,1,'hash'),
            layer_plan=[['int4','int4']],
            config=FlexGenPcacheConfig(Path('unused'),Path('unused'),4),
            method='impress',online_selection=True,keep_ratio=.5,
            logical_to_physical=mapping,physical_to_logical=inverse)
        mapping[0][0]=7;inverse[0][0]=7
        self.assertEqual(loader._storage_positions(0,[0,1,3]),[2,0,1])
        loader.configure_impress_layer(layer=0,selected_tokens=[3,1,0])
        selected=loader._selected_by_layer[0]
        first=loader._storage_positions(0,selected);first[0]=99
        self.assertEqual(loader._storage_positions(0,selected),[2,0,1])
        loader.configure_impress_layer(layer=0,selected_tokens=[7,6,5])
        self.assertEqual(loader._storage_positions(0,loader._selected_by_layer[0]),[5,6,4])
        self.assertEqual(loader._storage_positions(0,selected),[2,0,1])
        ids=torch.tensor([0,1,2,3,4,5,6,7]);k=torch.arange(8).view(8,1)
        a,b,c=loader._logical_segment(0,k,k+8,ids)
        self.assertEqual(c.tolist(),list(range(8)))
        self.assertEqual(a[:,0].tolist(),[2,0,3,1,7,5,6,4])

    def test_access_accounting_counts_unique_chunks_and_tail(self):
        layer=SimpleNamespace(chunk_num=3,device_map=['cpu','disk','cuda:0','disk','cpu','cuda:1'])
        loader=FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._pcache=SimpleNamespace(cache=[SimpleNamespace(layers=[layer])])
        loader._prefix_id=0;loader._config=SimpleNamespace(chunk_size=4)
        loader._info=SimpleNamespace(prefix_tokens=10)
        loader._source_tensor_tokens=dict(gpu=0,cpu=0,disk=0)
        loader._physical_tokens=0;loader._physical_chunks=set()
        loader._record_physical_access(0,np.array([0,1,1,4,8,9]))
        self.assertEqual(loader._source_tensor_tokens,dict(gpu=4,cpu=8,disk=8))
        self.assertEqual(loader._physical_tokens,6)
        self.assertEqual(loader._physical_chunks,{(0,0),(0,1),(0,2)})
        layer.device_map=['cpu']*6
        loader._record_physical_access(0,[8,9])
        self.assertEqual(loader._source_tensor_tokens,dict(gpu=4,cpu=12,disk=8))
        self.assertEqual(loader._physical_tokens,8)
