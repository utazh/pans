"""CPU algebra and mocked orchestration tests. Not Qwen/GPU/SSD certification."""
import copy,json,sys,types,itertools
from pathlib import Path
import numpy as np
import pytest
import torch
from contiguous_fuxian.sharedkv_v28 import common as C
from contiguous_fuxian.sharedkv_v28 import protocol as P
from contiguous_fuxian.sharedkv_v28 import execute as E
from contiguous_fuxian.sharedkv_v28.factors import svd_master,export_factor,representation
from contiguous_fuxian.sharedkv_v28.kvsharer import author_edges,author_proposal
from contiguous_fuxian.sharedkv_v28.analyze import paired_times
from contiguous_fuxian.sharedkv_v28.resources import reference_bytes
from contiguous_fuxian.sharedkv_v27.baseline_math import reconstruct,rotary
from contiguous_fuxian.sharedkv_v27.frozen_control import make_frozen_r0


def parent():
    r=dict(A=torch.eye(2).repeat(2,1,1),B=torch.eye(2).repeat(2,1,1),bk=torch.zeros(2,2),bv=torch.zeros(2,2))
    return dict(format='pans-shared-prefix-native-suffix-v25',native_policy=dict(blocks=4,chunk_size=16,rule='last_selected_blocks'),
      geometry=dict(layers=4,q_heads=4,kv_heads=2,head_dim=2),sources=[0,2,2,2],readers={'1':copy.deepcopy(r),'3':copy.deepcopy(r)},
      task='trec',token_hash='prefix',prefix_tokens=2,period=8,keep_ratio=.25,model_identity='model',calibration_query_hashes=[])


def manifest(tmp_path):
    spec=dict(repo=str(tmp_path),python=sys.executable,model_path=str(tmp_path/'model'),store_root=str(tmp_path/'store'),
       selector_index_dir=str(tmp_path/'index'),pcache_root=str(tmp_path/'cache'),gpu='0',blocks=6,block_size=2,warmup=1,
       conditions=['preserve','warm'],ranks=[[256,384],[384,576]],kvsharer_g=[24,20,16])
    cell=dict(id='trec_b25',task='trec',budget=25,expected_n=500,r4=str(tmp_path/'r4/trec.pt'),r0=str(tmp_path/'r0/trec.pt'))
    return dict(spec=spec,root=str(tmp_path/'study'),input_sha256={},cells=[cell]),cell


def test_frozen():
    b=parent();r=make_frozen_r0(b);P.assert_frozen_readers(b,r)
    assert b['native_policy']['blocks']==4 and r['native_policy']['blocks']==0

@pytest.mark.parametrize('field',['A','B','bk','bv'])
def test_reject_weight_change(field):
    b=parent();r=make_frozen_r0(b);r['readers']['1'][field].view(-1)[0]+=1
    with pytest.raises(ValueError):P.assert_frozen_readers(b,r)

@pytest.mark.parametrize('field',['sources','token_hash','calibration_query_hashes'])
def test_reject_metadata_change(field):
    b=parent();r=make_frozen_r0(b);r[field]=['changed'] if field!='token_hash' else 'changed'
    with pytest.raises(ValueError):P.assert_frozen_readers(b,r)

@pytest.mark.parametrize('rank',[1,3,8])
def test_svd_shape(rank):
    torch.manual_seed(12);x=[torch.randn(11,2,2),torch.randn(11,2,2)]
    master=svd_master(x);f=export_factor(master,rank,torch.float64)
    assert f['left'].shape==(11,rank)
    actual=reconstruct(f,1,torch.arange(11))
    target=master['u'][:,:rank]@torch.diag(master['s'][:rank])@master['vh'][:rank,4:]
    torch.testing.assert_close(actual.flatten(1),target)
    if rank==8:torch.testing.assert_close(actual,x[1].double(),atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_common_master_cast(dtype):
    torch.manual_seed(3);master=svd_master([torch.randn(10,2,2)]*2)
    f=export_factor(master,4,dtype)
    assert torch.equal(f['left'],master['u'][:,:4].to(dtype))
    assert torch.equal(f['rights'][0],(master['s'][:4,None]*master['vh'][:4,:4]).to(dtype))

def test_zero_singular_safe():
    master=svd_master([torch.zeros(10,2,2)]*2)
    for dtype in (torch.float32,torch.bfloat16):
        f=export_factor(master,4,dtype)
        assert torch.isfinite(f['left']).all() and torch.isfinite(f['rights'][0]).all()
        assert torch.count_nonzero(f['rights'][0])==0

@pytest.mark.parametrize('rank',[0,9,-1])
def test_rank_no_silent_clip(rank):
    with pytest.raises(ValueError):export_factor(svd_master([torch.randn(10,2,2)]*2),rank,torch.float32)

def test_rope_roundtrip_and_selected_rows():
    x=torch.randn(15,2,4,dtype=torch.float64);ids=torch.tensor([1,4,7,14]);pos=torch.arange(15)
    pre=rotary(x,pos,inverse=True);f=export_factor(svd_master([pre]),8,torch.float64)
    restored=rotary(reconstruct(f,0,ids),ids)
    torch.testing.assert_close(restored,x[ids],atol=1e-10,rtol=1e-10)

def test_author_distance_not_norm_of_mean():
    z=torch.zeros(2,1,2);a=torch.tensor([[[3.,4.]],[[3.,4.]]])
    edges=author_edges({0:(z,z),1:(a,-a)})
    assert edges==[(5.,0,1)]  # averaging K and V first would incorrectly give 0

@pytest.mark.parametrize('sources,s,t,expected',[
 ([0,1,2,3],0,2,[0,1,0,3]),([0,1,1,3],0,1,[0,0,0,3]),([0,1,2,2],1,2,[0,1,1,1])])
def test_author_remap(sources,s,t,expected):assert author_proposal(sources,s,t)==expected

@pytest.mark.parametrize('blocks',[3,6,9,12])
def test_order_balanced(blocks):
    orders=P.orders(blocks)
    for pos in range(3):
        assert all(sum(x[pos]==arm for x in orders)==blocks//3 for arm in ('I','R4','R0'))
    if blocks==6:assert set(orders)==set(itertools.permutations(('I','R4','R0')))

@pytest.mark.parametrize('blocks',[0,1,2,4,7])
def test_order_reject(blocks):
    with pytest.raises(ValueError):P.orders(blocks)

@pytest.mark.parametrize('arm',['I','R4','R0'])
def test_command_nesting(tmp_path,arm):
    m,c=manifest(tmp_path);j=P.command(m,c,arm,'timing',block=0,condition='preserve');a=j['argv']
    assert a.count('--')==4 and a[a.index('--admission')+1]=='reuse-prefetch'
    assert a[a.index('--cold')+1]=='ssd-cold' and a[a.index('--warmup')+1]=='1'
    assert '--diagnostic' not in a and '--selection-plan' not in a
    assert ('--adapters' in a)==(arm!='I')

def test_external_cannot_time(tmp_path):
    m,c=manifest(tmp_path)
    for kw in [dict(condition='preserve'),dict(block=0)]:
        with pytest.raises(ValueError):P.command(m,c,'clf','external_quality',representation='x.pt',**kw)
    j=P.command(m,c,'clf','external_quality',representation='x.pt')
    assert j['quality_only'] and '--diagnostic' in j['argv'] and '--cold' not in j['argv']
    assert j['argv'][1].endswith('run_baseline_quality_v27.py')

def test_exact_score_gate(tmp_path):
    for p,delta in [('a',0),('b',.01),('c',.5)]:
        C.write_jsonl(tmp_path/p/'scored_records.jsonl',[dict(uid=1,label_candidate_mean_logprobs={'yes':-1+delta,'no':-2})])
    assert E.score_parity(tmp_path/'a',tmp_path/'b')['passed']
    assert not E.score_parity(tmp_path/'a',tmp_path/'c')['passed']
    assert E.score_parity(tmp_path/'a',tmp_path/'c')['threshold']==.05


def test_timing_uses_blocks_not_requests():
    r=paired_times({0:100,1:200,2:300},{0:80,1:160,2:240})
    assert r['blocks']==3 and r['delta_percent']==pytest.approx(-20)
    assert not r['stable_speed_certified']

@pytest.mark.parametrize('a,b', [({0:1},{0:1}),({0:1,1:1,2:1},{0:1,1:1,3:1}),
                                   ({0:1,1:1,2:1},{0:1,1:0,2:1})])
def test_bad_timing(a,b):
    with pytest.raises(ValueError):paired_times(a,b)


def test_resource_union_counts_only_once():
    b=dict(method='direct-map-prefix-quality',sources=[0,0],native={'0':[torch.zeros(10,1,2),torch.zeros(10,1,2)]})
    a=reference_bytes(b,[[0,1,2],[2,3]])
    assert a['source_union_tokens']==4 and a['selected_rows_reference_bytes']==4*2*2*4


def test_factor_resource_separates_resident_rights():
    master=svd_master([torch.randn(10,1,2)]*2);fac=export_factor(master,2,torch.float32)
    b=dict(method='xKV-CLF-prefix-quality',groups=[dict(layers=[0,1],key=fac,value=fac)])
    a=reference_bytes(b,[[0,1],[1,2]])
    assert a['selected_rows_reference_bytes']==3*2*4*2
    assert a['rights_resident_bytes']==2*2*2*4*2


def test_exclusive_write_and_hash(tmp_path):
    p=tmp_path/'a.json';C.write_json(p,{'a':1});h=C.sha(p);C.assert_hashes({str(p):h})
    with pytest.raises(FileExistsError):C.write_json(p,{'a':2})
    p.write_text('changed')
    with pytest.raises(ValueError):C.assert_hashes({str(p):h})


def test_canonical_spec(tmp_path):
    s=P.make_spec(tmp_path,tmp_path/'model',tmp_path/'cache')
    assert len(s['cells'])==16
    for c in s['cells']:
        if c['task']=='subj':
            assert '_canonical/subj.pt' in c['r4'] and 'canonical_full_eval' in c['eval_file']
            assert c['expected_n']==1003
    t=next(c for c in s['cells'] if c['id']=='trec_b25')
    assert t['r4'].endswith('v25_g16_r4_s24/trec.pt')


def test_foreground_job_resume_and_hash(tmp_path,monkeypatch):
    m,c=manifest(tmp_path);j=P.command(m,c,'I','timing',block=0,condition='preserve')
    b=Path(j['argv'][j['argv'].index('--bundle-dir')+1])
    rows=[dict(uid=str(i),task='trec',correct=True,ttft_ms=10) for i in range(2)]
    C.write_jsonl(b/'trec.jsonl',rows)
    def fake(args,**kwargs):
        p=Path(j['output']);C.write_jsonl(p/'scored_records.jsonl',rows)
        C.write_jsonl(str(p)+'.cold.jsonl',[dict(status='completed',page_after={'resident':0})]*2)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(E.subprocess,'run',fake)
    x=E.run_job(m,j,True);assert x['status']=='complete'
    assert E.run_job(m,j,True)==x
    Path(j['output'],'scored_records.jsonl').write_text('')
    with pytest.raises(ValueError):E.run_job(m,j,True)


def test_cold_failure_not_promoted(tmp_path,monkeypatch):
    m,c=manifest(tmp_path);j=P.command(m,c,'I','timing',block=0,condition='preserve')
    rows=[dict(uid=str(i),task='trec',correct=True,ttft_ms=10) for i in range(2)]
    C.write_jsonl(Path(j['argv'][j['argv'].index('--bundle-dir')+1])/'trec.jsonl',rows)
    def fake(args,**kwargs):
        C.write_jsonl(Path(j['output'])/'scored_records.jsonl',rows)
        C.write_jsonl(j['output']+'.cold.jsonl',[dict(status='completed',page_after={'resident':1})]*2)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(E.subprocess,'run',fake)
    with pytest.raises(RuntimeError):E.run_job(m,j,True)
    state=json.loads((Path(m['root'])/'jobs'/(j['id']+'.json')).read_text())
    assert state['status']=='invalid_cold'

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable; no GPU numerical certification')
def test_cuda_svd_reference():
    x=[torch.randn(12,2,2,device='cuda')]*2
    f=export_factor(svd_master(x),8,torch.float64)
    torch.testing.assert_close(reconstruct(f,0,torch.arange(12)),x[0].cpu().double(),atol=1e-10,rtol=1e-10)


def toy_study(tmp_path,monkeypatch,leak=False):
    """Real on-disk record/checkpoint validation; synthetic tokenizer/model only."""
    from contiguous_fuxian.sharedkv_v24 import records as R
    tasks={k:8 for k in C.TASKS};monkeypatch.setattr(P,'TASKS',tasks);monkeypatch.setattr(C,'TASKS',tasks)
    monkeypatch.setattr(P,'check',lambda repo:None)  # this fixture is NOT a repository integrity test
    class Tokenizer:
        def __call__(self,text,**kwargs):return types.SimpleNamespace(input_ids=[int(x) for x in text.split()])
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    spec=P.make_spec(tmp_path,tmp_path/'model',tmp_path/'cache');spec.update(blocks=6,block_size=1,warmup=1,paths_are_templates_verify_before_prepare=False)
    for key in ('model_path','store_root','selector_index_dir','pcache_root'):Path(spec[key]).mkdir(parents=True,exist_ok=True)
    C.write_json(Path(spec['model_path'])/'config.json',{'toy':True});C.write_json(Path(spec['selector_index_dir'])/'manifest.json',{'toy':True})
    created=set()
    for cell in spec['cells']:
        task=cell['task'];b=parent();b.update(task=task,keep_ratio=cell['budget']/100)
        def record(q,role):
            plan=[[0,1]]*4
            return dict(format=R.FORMAT,**{k:b[k] for k in C.FIELDS},query_hash=R.digest_ids([q]),query_token_ids=[q],
               teacher_sources=list(range(4)),score_rule='mean_continuation_logprob',label_order=['a','b'],label_ids={'a':[1],'b':[2]},
               teacher_token_logprobs={'a':[-1.],'b':[-2.]},teacher_scores={'a':-1.,'b':-2.},
               selection_by_layer=plan,selection_hash=R.json_digest(plan),split=role)
        fs=[record(100,'fit')];vs=[record(x,'validation') for x in range(200,216)]
        for name,records in [('teacher_fit',fs),('teacher_val',vs)]:
            for row in records:C.save_tensor(Path(cell[name])/task/(row['query_hash']+'.pt'),row)
        b['calibration_query_hashes']=[x['query_hash'] for x in fs+vs]
        C.save_tensor(cell['r4'],b)
        if task not in created:
            C.write_json(Path(spec['store_root'])/task/'metadata.json',{'token_hash':'prefix'})
            def source_row(x):return dict(uid=str(x),task=task,prefix_text='1 2',query_text=str(x),answer='a')
            es=[source_row(x) for x in range(300,308)]
            if leak:es[0]=source_row(100)
            C.write_jsonl(cell['eval_file'],es);C.write_jsonl(cell['validation_file'],[source_row(x) for x in range(200,216)])
            created.add(task)
    return spec


def test_prepare_full_synthetic_matrix(tmp_path,monkeypatch):
    spec=toy_study(tmp_path,monkeypatch);folder=tmp_path/'study'
    m=P.prepare(spec,folder);assert len(m['cells'])==16
    for c in m['cells']:P.assert_frozen_readers(C.load_tensor(c['r4']),C.load_tensor(c['r0']))
    for task in C.TASKS:
        cell=next(c for c in m['cells'] if c['task']==task)
        rows=[r for b in range(6) for r in C.read_jsonl(folder/'bundles'/cell['id']/f'b{b:02d}'/(task+'.jsonl'))]
        assert len(rows)==len({r['uid'] for r in rows})==6
    C.assert_hashes(m['input_sha256'])


def test_prepare_reject_leak_before_writes(tmp_path,monkeypatch):
    spec=toy_study(tmp_path,monkeypatch,leak=True)
    with pytest.raises(ValueError,match='calibration bans'):P.prepare(spec,tmp_path/'study')
    assert not (tmp_path/'study').exists()


def test_prepare_requires_explicit_path_review(tmp_path,monkeypatch):
    spec=toy_study(tmp_path,monkeypatch);spec['paths_are_templates_verify_before_prepare']=True
    with pytest.raises(ValueError,match='Review all paths'):P.prepare(spec,tmp_path/'study')


def test_representation_exports_quality_flags():
    b=parent();b['geometry'].update(layers=2);b['prefix_tokens']=10
    f=svd_master([torch.randn(10,2,2)]*2)
    master=dict(**{k:b[k] for k in ('task','token_hash','model_identity','geometry','prefix_tokens')},
        rope_theta=1e6,groups=[dict(layers=[0,1],key=f,value=f)],lookup=[[0,0],[0,1]])
    r=representation(master,b,{'fit'},(3,4),torch.bfloat16)
    assert r['quality_only'] and not r['latency_valid'] and not r['hardware_bytes_valid']
    assert r['factor_precision']=='torch.bfloat16'


def test_quality_import_does_not_forge_ni():
    from contiguous_fuxian.sharedkv_v27.quality import paired_quality
    a={str(i):{'correct':True,'task':'trec'} for i in range(16)}
    p=paired_quality(a,copy.deepcopy(a));assert not p['certified'] and p['delta_pp']==0
