import copy,json,os,sys,threading,types,weakref
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from torch import nn

from contiguous_fuxian.sharedkv_v23.contracts import Group,source_map,validate_pair,TRACE
from contiguous_fuxian.sharedkv_v23.block import (Parameters,TrainableReader,segment,shared_attention,
    replay_group,boundary_loss,rope)
from contiguous_fuxian.sharedkv_v23.capture import GroupCollector,assemble
from contiguous_fuxian.sharedkv_v23 import cold
from contiguous_fuxian.sharedkv.core import Reader

torch.set_num_threads(1)

class RMS(nn.Module):
    def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6).to(x.dtype)*self.weight
class Rotary(nn.Module):
    def forward(self,x,positions):
        freq=torch.tensor([1.,.07],device=x.device);phase=positions[...,None]*freq
        phase=torch.cat([phase,phase],-1)
        return phase.cos().to(x.dtype),phase.sin().to(x.dtype)
class Attn(nn.Module):
    def __init__(self):
        super().__init__();self.config=NS(num_attention_heads=4,num_key_value_heads=2,hidden_size=16)
        self.head_dim=4;self.q_proj=nn.Linear(16,16);self.k_proj=nn.Linear(16,8)
        self.v_proj=nn.Linear(16,8);self.o_proj=nn.Linear(16,16,bias=False)
class Layer(nn.Module):
    def __init__(self):
        super().__init__();self.input_layernorm=RMS(16);self.self_attn=Attn()
        self.post_attention_layernorm=RMS(16);self.mlp=nn.Sequential(nn.Linear(16,24),nn.SiLU(),nn.Linear(24,16))
class Core(nn.Module):
    def __init__(self,layers=4):
        super().__init__();self.layers=nn.ModuleList([Layer() for _ in range(layers)])
        self.embed_tokens=nn.Embedding(32,16);self.norm=RMS(16);self.rotary_emb=Rotary()
class Model(nn.Module):
    def __init__(self,layers=4):
        super().__init__();self.model=Core(layers);self.lm_head=nn.Linear(16,32,bias=False)
        self.config=NS(model_type='qwen2',rope_scaling=None)

def identity(h=2,d=4,dtype=torch.float32):
    return Reader(torch.eye(d,dtype=dtype).repeat(h,1,1),torch.eye(d,dtype=dtype).repeat(h,1,1),
        torch.zeros(h,d,dtype=dtype),torch.zeros(h,d,dtype=dtype))

def rand_prefix(layers=4,n=7):
    return {l:(torch.randn(1,n,2,4)*.4,torch.randn(1,n,2,4)*.4) for l in range(layers)}

@pytest.mark.parametrize('size,count',[(2,14),(4,7),(8,4)])
def test_global_targets(size,count):
    groups=[Group(i,min(i+size,28),i) for i in range(0,28,size)]
    m=source_map(groups);assert len(set(m))==count;assert sum(i!=s for i,s in enumerate(m))==28-count

@pytest.mark.parametrize('group', [Group(7,10,7),Group(8,12,9),Group(2,4,8),Group(-1,2,0)])
def test_illegal_groups(group):
    with pytest.raises(ValueError):group.validate(28,8)

def test_many_consumers_not_pairs():
    m=source_map([Group(24,28,24)]);assert m[24:]==[24]*4;assert len(set(m))==25

def test_explicit_affine_equivalence():
    torch.manual_seed(3);q=torch.randn(1,3,4,4,dtype=torch.double)
    pk,pv=[torch.randn(1,7,2,4,dtype=torch.double) for _ in range(2)]
    tk,tv=[torch.randn(1,3,2,4,dtype=torch.double) for _ in range(2)]
    r=Parameters(torch.randn(2,4,4,dtype=torch.double),torch.randn(2,4,4,dtype=torch.double),
        torch.randn(2,4,dtype=torch.double),torch.randn(2,4,dtype=torch.double))
    k=torch.einsum('bnhd,hde->bnhe',pk,r.A)+r.bk[None,None]
    v=torch.einsum('bnhd,hde->bnhe',pv,r.B)+r.bv[None,None]
    expect,_=segment(q,torch.cat([k,tk],1),torch.cat([v,tv],1),True)
    got,_=shared_attention(q,pk,pv,tk,tv,r)
    torch.testing.assert_close(got,expect,rtol=1e-11,atol=1e-11)

def test_joint_normalizer_and_bias():
    q=torch.ones(1,1,2,4);k=torch.zeros(1,1,1,4);v=torch.ones_like(k)
    tailv=torch.zeros_like(k);r=Parameters(torch.eye(4)[None],torch.eye(4)[None],torch.ones(1,4)*4,torch.zeros(1,4))
    out,b=shared_attention(q,k,v,k,tailv,r)
    assert b.min()>.99 and out.min()>.99

def test_group_backward_only_readers():
    torch.manual_seed(8);m=Model().eval().requires_grad_(False);g=Group(0,4,0)
    x=torch.randn(1,5,16);pos=torch.arange(7,12)[None];kv=rand_prefix()
    readers=nn.ModuleDict({str(i):TrainableReader(identity(),ste=False) for i in g.consumers})
    out,_,_=replay_group(m,g,x,pos,kv,readers)
    out.square().mean().backward()
    for r in readers.values():
        assert r.B.grad is not None and r.B.grad.abs().sum()>0
        assert r.A.grad is not None and r.A.grad.abs().sum()>0
    assert all(p.grad is None for p in m.parameters())

def test_multiple_consumers_receive_same_prefix_object(monkeypatch):
    import contiguous_fuxian.sharedkv_v23.block as b
    seen=[];old=b.layer_forward
    def wrapped(layer,x,pk,pv,*args,**kwargs):
        seen.append((pk.data_ptr(),pv.data_ptr()));return old(layer,x,pk,pv,*args,**kwargs)
    monkeypatch.setattr(b,'layer_forward',wrapped)
    model=Model().eval().requires_grad_(False);g=Group(0,4,0);p=rand_prefix()
    readers=nn.ModuleDict({str(i):TrainableReader(identity(),False) for i in g.consumers})
    replay_group(model,g,torch.randn(1,3,16),torch.arange(3)[None],p,readers)
    assert len(set(seen))==1

def test_suffix_weight_has_gradient():
    t=torch.randn(1,5,8);s=(t+.1).requires_grad_();h=torch.zeros_like(t)
    loss,_=boundary_loss(s,t,h,3,last_weight=0,suffix_weight=4)
    loss.backward();assert s.grad[:,3:].abs().sum()>s.grad[:,:3].abs().sum()

def test_joint_optimization_synthetic():
    torch.manual_seed(11);model=Model(3).eval().requires_grad_(False);g=Group(0,3,0)
    x=torch.randn(1,4,16);pos=torch.arange(7,11)[None];kv=rand_prefix(3)
    true=nn.ModuleDict({str(i):TrainableReader(identity(),False) for i in g.consumers})
    with torch.no_grad():
        true['1'].B.mul_(1.3);true['2'].bv.add_(.2)
        target=replay_group(model,g,x,pos,kv,true)[0]
    learn=nn.ModuleDict({str(i):TrainableReader(identity(),False) for i in g.consumers})
    opt=torch.optim.Adam(learn.parameters(),lr=.01)
    errors=[]
    for _ in range(35):
        y=replay_group(model,g,x,pos,kv,learn)[0];loss=(y-target).square().mean();errors.append(float(loss.detach()))
        opt.zero_grad();loss.backward();opt.step()
    assert errors[-1]<errors[0]*.2
    assert all(p.grad is None for p in model.parameters())

def make_runner(model,kv,query=(1,2,3),suffixes=((6,7),(6,8),(9,))):
    runner=NS()
    def decoder(*,model,input_ids,cache,position_start,loader,**kwargs):
        x=model.model.embed_tokens(input_ids);b,t,_=x.shape
        cos,sin=model.model.rotary_emb(x,torch.arange(position_start,position_start+t)[None])
        for lid,layer in enumerate(model.model.layers):
            z=layer.input_layernorm(x);a=layer.self_attn
            q=rope(a.q_proj(z).reshape(b,t,4,4),cos,sin)
            k=rope(a.k_proj(z).reshape(b,t,2,4),cos,sin);v=a.v_proj(z).reshape(b,t,2,4)
            if lid in cache:k,v=(torch.cat((cache[lid][0],k),1),torch.cat((cache[lid][1],v),1))
            cache[lid]=(k,v)
            out,_=shared_attention(q,*kv[lid],k,v)
            mid=x+a.o_proj(out.reshape(b,t,16));x=mid+layer.mlp(layer.post_attention_layernorm(mid))
        return model.lm_head(model.model.norm(x))
    runner.flexgen_sparse_decoder_logits=decoder
    def completion(*args,**kwargs):
        cache={};loader=kwargs['loader'];ids=kwargs['query_token_ids'];n=loader.prefix_tokens
        runner.flexgen_sparse_decoder_logits(model=model,input_ids=torch.tensor([ids]),cache=cache,position_start=n,loader=loader)
        for suffix in suffixes:
            cc={l:tuple(t.clone() for t in v) for l,v in cache.items()}
            for j,tok in enumerate(suffix):
                runner.flexgen_sparse_decoder_logits(model=model,input_ids=torch.tensor([[tok]]),cache=cc,
                    position_start=n+len(ids)+j,loader=loader)
        return 'x',1.,2.,{}
    runner.greedy_flexgen_completion=completion
    return runner

def fake_loader(layers=4,n=7):
    return NS(_info=NS(task='trec',token_hash='hash'),prefix_tokens=n,keep_ratio=1.,
       _selector_index=NS(payload={'model_identity':{'synthetic':True}}),
       _selected_by_layer=[list(range(n)) for _ in range(layers)])

def fake_runtime(layers=4):
    g={'layers':layers,'q_heads':4,'kv_heads':2,'head_dim':4}
    return NS(_prepare=lambda *a:(g,list(range(layers)),{},set()))

def test_full_capture_matches_causal_group_replay(tmp_path):
    torch.manual_seed(2);m=Model().eval();kv=rand_prefix();runner=make_runner(m,kv)
    GroupCollector(tmp_path,0,4,2).install(fake_runtime(),runner)
    with torch.inference_mode():runner.greedy_flexgen_completion(model=m,loader=fake_loader(),max_tokens=1,
        impress_selection_period_size=4,query_token_ids=[1,2,3])
    r=torch.load(next((tmp_path/'trec').glob('*.pt')),weights_only=True)
    assert len(r['variants'])==4
    for v in r['variants']:
        x=v['layer_inputs'][0];positions=torch.arange(7,7+x.shape[1])[None]
        out,_,_=replay_group(m,Group(0,4,0),x,positions,kv,{},native=True)
        torch.testing.assert_close(out,v['layer_outputs'][3],rtol=2e-5,atol=2e-6)
    assert (tmp_path/'selection').is_dir()

def test_capture_no_free_generation(tmp_path):
    m=Model();r=make_runner(m,rand_prefix());GroupCollector(tmp_path,0,4).install(fake_runtime(),r)
    with pytest.raises(ValueError,match='max_tokens'):
        r.greedy_flexgen_completion(model=m,loader=fake_loader(),max_tokens=2,impress_selection_period_size=4,query_token_ids=[1,2])

@pytest.mark.parametrize('key',['query_hash','query_token_ids','period','selection_by_layer'])
def test_pair_identity_rejects(key):
    row={k:'same' for k in ('task','token_hash','prefix_tokens','geometry','period','keep_ratio','model_identity')}
    row.update(query_hash='q',query_token_ids=[1],group_start=0,group_end=2,
        selection_by_layer=[[1],[1]],variants=[{'suffix_ids':[]}])
    other=copy.deepcopy(row);other[key]='bad'
    with pytest.raises(ValueError):validate_pair(row,other,Group(0,2,0))

def test_cold_rejects_external_path(tmp_path):
    owned=tmp_path/'owned';owned.mkdir();p=tmp_path/'x';p.write_bytes(b'123')
    with pytest.raises(ValueError):cold.safe_file(p,owned)

def test_cold_rejects_hardlink(tmp_path):
    p=tmp_path/'a';p.write_bytes(b'12');q=tmp_path/'b';os.link(p,q)
    with pytest.raises(ValueError):cold.safe_file(p,tmp_path)

def test_cold_rejects_symlink(tmp_path):
    p=tmp_path/'a';p.write_bytes(b'12');q=tmp_path/'b';q.symlink_to(p)
    with pytest.raises(ValueError):cold.safe_file(q,tmp_path)

def test_cold_fail_closed_if_pages_stay(tmp_path,monkeypatch):
    p=tmp_path/'x';p.write_bytes(b'1'*8192)
    monkeypatch.setattr(cold,'page_residency',lambda p:{'pages':2,'resident':1,'bytes':8192})
    monkeypatch.setattr(os,'posix_fadvise',lambda *a:None)
    with pytest.raises(RuntimeError,match='precondition FAILED'):cold.prepare_pages([p],tmp_path,'ssd-cold')

def test_app_cold_does_not_claim_ssd(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'1')
    r=cold.prepare_pages([p],tmp_path,'app-cold');assert r['page_after'] is None

@pytest.mark.skipif(sys.platform!='linux',reason='Linux only')
def test_mincore_real_owned_file(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'a'*8192)
    r=cold.page_residency(p);assert r['pages']==2 and 0<=r['resident']<=2

@pytest.mark.skipif(sys.platform!='linux',reason='Linux only')
def test_ssd_cold_file_eviction_live(tmp_path):
    p=tmp_path/'x'
    with p.open('wb') as f:f.write(b'a'*32768);f.flush();os.fsync(f.fileno())
    try:r=cold.prepare_pages([p],tmp_path,'ssd-cold')
    except RuntimeError as e:pytest.skip('Host filesystem refuses DONTNEED: '+str(e))
    assert r['page_after']['resident']==0
    assert p.read_bytes()==b'a'*32768 # Test data only; not part of performance path.

class ScoreUnit:
    def __init__(self,items):
        self.cache_type='CKLFU';self.gpu_num=len(items);self.cpu_num=0;self.gpu=list(items);self.cpu=[]
    def evict_gpu(self):
        x=self.gpu.pop();x.device='cpu';x.token.device='cpu';self.cpu.append(x);self.gpu_num-=1;self.cpu_num+=1
    def evict_cpu(self):
        x=self.cpu.pop();x.device='disk';x.token.device='disk';x.token.key=x.token.value=None
        x.layer.device_map[x.idx]='disk';self.cpu_num-=1
class Executor:
    def submit(self,*a,**kw):return NS(result=lambda **kw:None)

def fake_pcache(tmp_path):
    layer=NS(disk_type='KV_Division',disk_map=None,device_map=['cuda:0','cuda:0'],head_token=None)
    items=[]
    for i in range(2):
        p=tmp_path/f'{i}.npy';p.write_bytes(b'a'*4096)
        token=NS(path=str(p),file_exist=True,key=torch.ones(1),value=None,device='cuda:0')
        items.append(NS(token=token,device='cuda:0',control=NS(score=[5,5]),layer=layer,idx=i))
    layer.key_tokens=[items[0]];layer.value_tokens=[items[1]]
    pcache=NS(cache=[NS(layers=[layer])],control=ScoreUnit(items),_prefetch_executor=Executor(),
        _cache_lock=threading.RLock(),prefetch_scheduler_metrics=lambda:{})
    class Owner:pass
    owner=Owner();owner._attention_state={1:1};owner._impress_score_state={1:2}
    loader=NS(_pcache=pcache,_prefix_id=0,_v23_store_owner=weakref.ref(owner))
    return loader,owner

def test_controller_eviction_not_tensor_clear(tmp_path,monkeypatch):
    loader,owner=fake_pcache(tmp_path);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    paths,audit=cold.evict_application(loader,tmp_path)
    assert len(paths)==2 and audit['application_payload_empty']
    assert not owner._attention_state and loader._pcache.control.gpu_num==0
    assert loader._pcache.cache[0].layers[0].device_map==['disk','disk']

def test_cold_refuses_pending(tmp_path,monkeypatch):
    loader,owner=fake_pcache(tmp_path);loader._pending={0:object()}
    with pytest.raises(RuntimeError,match='too late'):cold.evict_application(loader,tmp_path)

def test_cold_refuses_unpersisted(tmp_path,monkeypatch):
    loader,owner=fake_pcache(tmp_path);loader._pcache.cache[0].layers[0].key_tokens[0].token.file_exist=False
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    with pytest.raises(RuntimeError,match='write new'):cold.evict_application(loader,tmp_path)

def test_cold_keeps_label_calls_out_of_io(tmp_path,monkeypatch):
    monkeypatch.setattr(cold,'evict_application',lambda *a:([],{'application_payload_empty':True}))
    monkeypatch.setattr(cold,'prepare_pages',lambda *a:{'page_after':{'resident':0}})
    count=[0]
    def io():count[0]+=1;return {'read_bytes':count[0]*4096}
    monkeypatch.setattr(cold,'io_snapshot',io)
    r=NS(flexgen_sparse_decoder_logits=lambda *a,**kw:None)
    def old(*a,**kw):
        for _ in range(4):r.flexgen_sparse_decoder_logits()
        return 'x',12.,14.,{}
    r.greedy_flexgen_completion=old
    ctl=cold.ColdController('ssd-cold',tmp_path,tmp_path/'cold.jsonl',1);ctl.install(r)
    kw={'loader':fake_loader(),'query_token_ids':[1,2,3]}
    r.greedy_flexgen_completion(**kw);r.greedy_flexgen_completion(**kw)
    assert count[0]==2
    record=json.loads((tmp_path/'cold.jsonl').read_text());assert record['io_first_decoder']['read_bytes']==4096

@pytest.mark.skipif(not torch.cuda.is_available(),reason='No CUDA device')
def test_cuda_reader_backward():
    m=Model().cuda().bfloat16().eval().requires_grad_(False);g=Group(0,4,0)
    kv={l:tuple(v.cuda().bfloat16() for v in p) for l,p in rand_prefix().items()}
    rs=nn.ModuleDict({str(i):TrainableReader(identity(),True) for i in g.consumers}).cuda()
    out,_,_=replay_group(m,g,torch.randn(1,5,16,device='cuda',dtype=torch.bfloat16),torch.arange(5,device='cuda')[None],kv,rs)
    out.float().square().mean().backward();assert all(r.B.grad is not None for r in rs.values())

def test_group_train_cli_synthetic_closed_loop(tmp_path,monkeypatch):
    from contiguous_fuxian.sharedkv_v23.train import main
    torch.manual_seed(21);model=Model().eval();kv=rand_prefix()
    kv={l:tuple(x.bfloat16().half().bfloat16().float() for x in pair) for l,pair in kv.items()}
    store=tmp_path/'store'/'trec';store.mkdir(parents=True)
    (store/'metadata.json').write_text(json.dumps(dict(task='trec',token_hash='hash',dtype='bfloat16',
        layout='token,kv_head,head_dim',prefix_tokens=7,kv_heads=2,head_dim=4,layers=4)))
    for lid,pair in kv.items():
        for kind,x in zip(('key','value'),pair):
            x[0].contiguous().bfloat16().view(torch.uint16).numpy().tofile(store/f'layer_{lid:02d}_{kind}.bf16')
    for name,queries in [('fit',([1,2,3],[2,3,4])),('val',([3,4,5],[4,5,6]))]:
        for query in queries:
            runner=make_runner(model,kv,query,suffixes=((6,7),))
            GroupCollector(tmp_path/name,0,4,2).install(fake_runtime(),runner)
            with torch.inference_mode():runner.greedy_flexgen_completion(model=model,loader=fake_loader(),max_tokens=1,
                impress_selection_period_size=4,query_token_ids=list(query))
    transf=types.ModuleType('transformers')
    transf.AutoModelForCausalLM=NS(from_pretrained=lambda *a,**kw:copy.deepcopy(model))
    monkeypatch.setitem(sys.modules,'transformers',transf)
    ident=types.ModuleType('contiguous_fuxian.quantized_key_index');ident.model_identity=lambda _: {'synthetic':True}
    monkeypatch.setitem(sys.modules,'contiguous_fuxian.quantized_key_index',ident)
    output=tmp_path/'out'/'trec.pt'
    main(['--model-path',str(tmp_path/'mock_model'),'--store-root',str(tmp_path/'store'),'--task','trec',
          '--teacher-fit',str(tmp_path/'fit'),'--student-fit',str(tmp_path/'fit'),
          '--teacher-val',str(tmp_path/'val'),'--student-val',str(tmp_path/'val'),
          '--start','0','--end','4','--source','0','--device','cpu','--steps','2','--eval-every','1',
          '--logit-weight','.05','--output',str(output)])
    bundle=torch.load(output,weights_only=True)
    assert bundle['sources']==[0,0,0,0] and len(bundle['readers'])==3
    assert bundle['diagnostic_only'] and not bundle['end_to_end_validated']
    assert len(bundle['calibration_query_hashes'])==4
    assert bundle['report']['suffix_group_update_relative_rmse'] is not None
    assert all(not p.requires_grad or p.grad is None for p in model.parameters())

def test_native_exception_keeps_leader_and_many_consumers():
    group=Group(24,28,26,(24,));group.validate(28,8)
    assert group.consumers==[25,27]
    sources=source_map([group]);assert sources[24:]==[24,26,26,26]

def test_budget_compatible_global14_template():
    groups=[Group(i,i+4,i+1,(i,)) for i in range(0,28,4)]
    mapping=source_map(groups);assert len(set(mapping))==14
    assert all(mapping[i]==i for i in (0,8,16,24))

def test_actual_native_exception_no_reader_needed():
    model=Model().eval().requires_grad_(False);group=Group(0,4,2,(0,))
    readers=nn.ModuleDict({str(i):TrainableReader(identity(),False) for i in group.consumers})
    out,_,_=replay_group(model,group,torch.randn(1,3,16),torch.arange(3)[None],rand_prefix(),readers)
    out.square().mean().backward();assert set(readers)=={'1','3'}


def test_scoped_read_advice_affects_worker_and_restores(tmp_path):
    p=tmp_path/'target';other=tmp_path/'other';p.write_bytes(b'a');other.write_bytes(b'b')
    calls=[]
    def native(path):calls.append(str(path))
    module=NS(uncache=native)
    with cold.RuntimeReadAdvice(module,[p]) as advice:
        thread=threading.Thread(target=lambda:module.uncache(p));thread.start();thread.join()
        module.uncache(other)
        assert advice.suppressed==1 and calls==[str(other)]
    assert module.uncache is native
    module.uncache(p);assert calls[-1]==str(p)


def test_read_advice_restores_on_error(tmp_path):
    original=lambda path:None;module=NS(uncache=original)
    with pytest.raises(RuntimeError):
        with cold.RuntimeReadAdvice(module,[tmp_path/'x']):raise RuntimeError('test')
    assert module.uncache is original


def test_page_hot_rejects_native_eviction_policy(tmp_path):
    with pytest.raises(ValueError,match='requires preserve'):
        cold.ColdController('page-hot',tmp_path,tmp_path/'log',1)
