import dataclasses,json,random
from pathlib import Path
import numpy as np
import pytest
from contiguous_fuxian.promixed import select_promixed_gqa_blocks as select
from contiguous_fuxian.quantized_key_index import source_task_mapping, QuantizedKeyIndex, INDEX_FORMAT, file_sha256

def test_prepared_matches_direct():
    rng=random.Random(910)
    for _ in range(300):
        rows=[[rng.randrange(7) for _ in range(32)] for _ in range(4)]
        ranks=[sorted(range(32),key=lambda b:(-row[b],b)) for row in rows]
        opts=dict(keep_blocks=rng.randint(1,32),coverage_fraction=rng.choice([0.,0.1,0.5,1.]),reuse_mode=rng.choice(['fixed','adaptive']))
        assert dataclasses.asdict(select(rows,**opts))==dataclasses.asdict(select(rows,ranked_blocks=ranks,**opts))

@pytest.mark.parametrize('ranks', [[[0,0]],[[0,1]],[[1]],[[True,0]]])
def test_prepared_rejects_stale_or_invalid(ranks):
    with pytest.raises(ValueError): select([[1,2]],keep_blocks=1,ranked_blocks=ranks)

def test_subset_keeps_original_prefix_id(tmp_path):
    (tmp_path/'.contiguous_fuxian_complete').write_text(json.dumps({'registered_store_tasks':['sst2','subj','trec','rte']}))
    assert source_task_mapping(tmp_path)['trec']==2
    with pytest.raises(ValueError):source_task_mapping(tmp_path,['trec'])

def test_legacy_mapping_never_guessed(tmp_path):
    with pytest.raises(ValueError):source_task_mapping(tmp_path)
    assert source_task_mapping(tmp_path,['sst2','subj','trec','rte'])['trec']==2

def test_identity_rejects_same_shape_other_prefix_and_corruption(tmp_path):
    td=tmp_path/'trec';td.mkdir()
    (td/'layer_00.codes.u8').write_bytes(bytes(4))
    (td/'layer_00.scales.f16').write_bytes(bytes(4))
    entry=dict(directory='trec',prefix_tokens=2,layers=1,kv_heads=1,head_dim=4,token_hash='prefix-a',
               files_sha256={p.name:file_sha256(p) for p in td.iterdir()})
    (tmp_path/'manifest.json').write_text(json.dumps(dict(format=INDEX_FORMAT,bits=4,selector_kv_head_ids=[0],tasks={'trec':entry},model_identity={'model':'a'})))
    index=QuantizedKeyIndex(tmp_path)
    opts=dict(prefix_tokens=2,layers=1,kv_heads=1,head_dim=4)
    index.validate_task('trec',token_hash='prefix-a',**opts)
    with pytest.raises(ValueError,match='token identity'):index.validate_task('trec',token_hash='prefix-b',**opts)
    with pytest.raises(ValueError,match='model'):index.validate_model({'model':'b'})
    (td/'layer_00.codes.u8').write_bytes(bytes([1,0,0,0]))
    with pytest.raises(ValueError,match='digest'):index.validate_task('trec',token_hash='prefix-a',**opts)

def test_builder_subset_uses_registered_id_and_checks_content(tmp_path):
    from contiguous_fuxian.quantized_key_index import build_quantized_key_index
    source=tmp_path/'pcache';source.mkdir()
    store=tmp_path/'store';(store/'trec').mkdir(parents=True)
    (source/'.contiguous_fuxian_complete').write_text(json.dumps({'registered_store_tasks':['sst2','subj','trec'],'selector_kv_head_ids':[0]}))
    meta=dict(task='trec',prefix_tokens=2,kv_heads=1,head_dim=4,layers=1,token_hash='trec-prefix',dtype='bfloat16',layout='token,kv_head,head_dim')
    (store/'trec'/'metadata.json').write_text(json.dumps(meta))
    values=np.full((2,1,4),2.,dtype=np.float32)
    (values.view(np.uint32)>>16).astype(np.uint16).tofile(store/'trec'/'layer_00_key.bf16')
    values.astype(np.float16).tofile(source/'2_0_None_k.npy')
    np.zeros_like(values,dtype=np.float16).tofile(source/'0_0_None_k.npy')
    args=dict(source_pcache_dir=source,store_root=store,tasks=['trec'],selector_kv_head_ids=[0],group_size=2)
    result=build_quantized_key_index(**args,output_dir=tmp_path/'index')
    assert result['tasks']['trec']['prefix_id']==2
    assert result['tasks']['trec']['token_hash']=='trec-prefix'
    np.zeros_like(values,dtype=np.float16).tofile(source/'2_0_None_k.npy')
    with pytest.raises(ValueError,match='content/mapping'):
        build_quantized_key_index(**args,output_dir=tmp_path/'invalid')

def test_runtime_checks_bundle_against_store_identity(tmp_path):
    from types import SimpleNamespace
    from contiguous_fuxian.flexgen_qwen_reprefill import validate_bundle_prefix_identity
    from contiguous_fuxian.sparse_qwen_reprefill import _token_hash
    task_dir=tmp_path/'trec';task_dir.mkdir()
    payload=dict(task='trec',prefix_tokens=2,layers=1,kv_heads=1,head_dim=4,token_hash=_token_hash([1,2]))
    (task_dir/'metadata.json').write_text(json.dumps(payload))
    def tokenizer(text,**kwargs):
        return SimpleNamespace(input_ids={'original':[1,2],'same-length-other':[2,1]}[text])
    rows=[dict(task='trec',prefix_text='original')]
    assert validate_bundle_prefix_identity(rows,store_root=tmp_path,tokenizer=tokenizer)=={'trec':payload['token_hash']}
    with pytest.raises(ValueError,match='identity mismatch'):
        validate_bundle_prefix_identity([dict(task='trec',prefix_text='same-length-other')],store_root=tmp_path,tokenizer=tokenizer)
    with pytest.raises(ValueError,match='one shared prefix'):
        validate_bundle_prefix_identity(rows+[dict(task='trec',prefix_text='same-length-other')],store_root=tmp_path,tokenizer=tokenizer)
