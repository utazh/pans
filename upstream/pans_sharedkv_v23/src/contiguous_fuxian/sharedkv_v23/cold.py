"""Linux payload-cold preparation; no global drop_caches, no O_DIRECT rewrite.

Eviction/setup is outside original logits-ready TTFT. Selector index, readers,
model, metadata, allocator and kernels stay warm. 'ssd-cold' means application
payload empty + zero target-file resident pages at a pre-request mincore
snapshot. It does NOT assert SSD controller/NAND cache state or physical I/O
bytes. Files must be dedicated immutable benchmark copies owned by this user.
"""
from __future__ import annotations
from contextvars import ContextVar
from contextlib import nullcontext
import ctypes,errno,gc,json,math,os,resource,stat,sys,time,weakref,threading
from pathlib import Path
import torch

_ACTIVE=ContextVar('v23_cold_request',default=None)


def page_residency(path):
    if sys.platform!='linux':raise RuntimeError('Linux mincore required')
    st=os.stat(path)
    if st.st_size==0:return {'pages':0,'resident':0,'bytes':0}
    # Linux can mask mincore for unowned, non-writable files; refuse such input.
    if st.st_uid!=os.geteuid():raise PermissionError('mincore audit requires file ownership')
    fd=os.open(path,os.O_RDONLY)
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mmap.restype=ctypes.c_void_p
    libc.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_longlong]
    libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_ubyte)]
    libc.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    ptr=libc.mmap(None,st.st_size,1,1,fd,0) # PROT_READ, MAP_SHARED; no data touch
    if ptr==ctypes.c_void_p(-1).value:
        error=ctypes.get_errno();os.close(fd);raise OSError(error,os.strerror(error))
    pages=(st.st_size+os.sysconf('SC_PAGE_SIZE')-1)//os.sysconf('SC_PAGE_SIZE')
    vec=(ctypes.c_ubyte*pages)()
    try:
        if libc.mincore(ptr,st.st_size,vec)!=0:
            e=ctypes.get_errno();raise OSError(e,os.strerror(e))
        return {'pages':pages,'resident':sum(v&1 for v in vec),'bytes':st.st_size}
    finally:
        libc.munmap(ptr,st.st_size);os.close(fd)


def safe_file(path,root):
    original=Path(path)
    if original.is_symlink():raise ValueError('Symlinked cache files not accepted')
    p=original.resolve(strict=True);p.relative_to(Path(root).resolve(strict=True))
    st=p.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid!=os.geteuid() or st.st_nlink!=1:
        raise ValueError('Use regular, owned, non-hardlinked dedicated benchmark cache copies')
    return p,(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)


def snapshot(paths):
    result={'files':len(paths),'pages':0,'resident':0,'bytes':0}
    for path in paths:
        r=page_residency(path)
        for k in ('pages','resident','bytes'):result[k]+=r[k]
    return result


def prepare_pages(paths,root,mode):
    if mode not in ('app-cold','page-hot','ssd-cold'):raise ValueError('Invalid cold mode')
    unique=sorted(set(str(Path(x)) for x in paths)); checked=[safe_file(x,root) for x in unique]
    before=snapshot([p for p,_ in checked]) if mode=='ssd-cold' else None
    for path,_ in checked:
        if mode=='ssd-cold':
            fd=os.open(path,os.O_RDONLY)
            try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
        elif mode=='page-hot':
            with path.open('rb',buffering=0) as f:
                while f.read(1024*1024):pass
    after=snapshot([p for p,_ in checked]) if mode!='app-cold' else None
    for p,identity in checked:
        if safe_file(p,root)[1]!=identity:raise RuntimeError('Cache file mutated during preparation')
    if mode=='ssd-cold' and after['resident']!=0:
        raise RuntimeError(f"Cold precondition FAILED: {after['resident']}/{after['pages']} pages remain resident; no timed trial")
    if mode=='page-hot' and after['resident']!=after['pages']:
        raise RuntimeError('Page-hot precondition failed under memory pressure')
    return {'page_before':before,'page_after':after,'mode':mode,
            'files_checked':len(checked),'file_bytes':sum(i[2] for _,i in checked)}


def io_snapshot():
    out={}
    for line in Path('/proc/self/io').read_text().splitlines():
        k,v=line.split(':');out[k]=int(v)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    out['major_faults']=usage.ru_majflt;out['minor_faults']=usage.ru_minflt
    return out


def delta(before,after):
    return {k:after[k]-before[k] for k in before}


def payload_items(pcache):
    all_items=[]
    for prefix in pcache.cache:
        for layer in prefix.layers:
            if getattr(layer,'disk_type',None)!='KV_Division':raise ValueError('Only audited KV_Division layout')
            if getattr(layer,'disk_map',None) is not None:raise ValueError('Unrecognized batched disk buffer')
            all_items.extend(list(layer.key_tokens));all_items.extend(list(layer.value_tokens))
            if getattr(layer,'head_token',None) is not None:all_items.append(layer.head_token)
    return all_items


def evict_application(loader,root):
    """Use real controller eviction; NEVER clear tensors behind a live heap."""
    pcache=loader._pcache
    for n in ('_pending','_missing_pending','_speculative_pending','_resolved_speculative','_loaded'):
        if getattr(loader,n,{}):raise RuntimeError('New request already has payload work; cold hook is too late')
    pcache._prefetch_executor.submit(lambda:None,priority=1e12,label='v23_cold_barrier').result(timeout=120)
    torch.cuda.synchronize()
    all_items=payload_items(pcache)
    target=[]
    for layer in pcache.cache[loader._prefix_id].layers:
        target.extend(list(layer.key_tokens));target.extend(list(layer.value_tokens))
    # Verify all eviction targets have committed files BEFORE calling to_disk.
    for item in all_items:
        token=item.token
        if not getattr(token,'file_exist',False):raise RuntimeError('Eviction would write new payload: only persisted caches supported')
        if not Path(token.path).is_file():raise FileNotFoundError(token.path)
    for item in target:safe_file(item.token.path,root)
    controller=pcache.control
    if type(controller).__name__!='ScoreUnit' or controller.cache_type!='CKLFU':
        raise ValueError('Only audited CKLFU ScoreUnit reset supported')
    limit=3*len(all_items)+1;count=0
    with pcache._cache_lock:
        while float(controller.gpu_num)>1e-8:
            controller.evict_gpu();count+=1
            if count>limit:raise RuntimeError('GPU heap did not drain')
        while float(controller.cpu_num)>1e-8:
            controller.evict_cpu();count+=1
            if count>limit:raise RuntimeError('CPU heap did not drain')
        for item in all_items:
            if str(item.device)!='disk':raise RuntimeError('Controller counters disagree with token residency')
            if getattr(item.token,'key',None) is not None or getattr(item.token,'value',None) is not None:
                raise RuntimeError('Payload tensor remains after controller eviction')
            item.control.score=[0,0]
        for prefix in pcache.cache:
            for layer in prefix.layers:
                # Ephemeral gather buffers, not payload ownership. No live request.
                layer.key_buffer=None;layer.value_buffer=None
                if any(str(x)!='disk' for x in layer.device_map):raise RuntimeError('device_map still warm')
    owner_ref=getattr(loader,'_v23_store_owner',None)
    owner=owner_ref() if owner_ref else None
    if owner is None:raise RuntimeError('Missing audited store ownership; cannot reset cumulative scores')
    owner._attention_state.clear();owner._impress_score_state.clear()
    torch.cuda.synchronize();gc.collect()
    # Exclude maintenance/barrier jobs from the request's cumulative scheduler delta.
    loader._prefetch_scheduler_start=pcache.prefetch_scheduler_metrics()
    return [item.token.path for item in target],{'payload_tokens_reset':len(all_items),
        'target_payload_chunks_k_plus_v':len(target),'application_payload_empty':True,
        'gpu_mib_after':float(controller.gpu_num),'cpu_mib_after':float(controller.cpu_num)}


def install_owner_hook():
    from contiguous_fuxian.flexgen_pcache import FlexGenPcacheStore
    original=FlexGenPcacheStore.new_layer_loader
    def new_loader(self,*args,**kwargs):
        loader=original(self,*args,**kwargs);loader._v23_store_owner=weakref.ref(self);return loader
    FlexGenPcacheStore.new_layer_loader=new_loader


class RuntimeReadAdvice:
    """Scope native uncache suppression to audited target files, including worker threads.

    The published vendor calls DONTNEED at EVERY disk read. Keeping that would
    destroy a page-hot control. Use the same explicit 'preserve' policy in both
    page-hot and ssd-cold controls; report native-policy results separately.
    """
    def __init__(self,module,paths):
        self.module=module
        self.paths={os.path.abspath(os.fspath(p)) for p in paths}
        self.suppressed=0;self._lock=threading.Lock()
    def __enter__(self):
        self.original=self.module.uncache
        def uncache(path):
            if os.path.abspath(os.fspath(path)) in self.paths:
                with self._lock:self.suppressed+=1
                return None
            return self.original(path)
        self.wrapper=uncache;self.module.uncache=uncache
        return self
    def __exit__(self,*exc):
        self.module.uncache=self.original


def read_advice_context(loader,paths,policy):
    if policy=='native':return nullcontext(None)
    if policy!='preserve':raise ValueError('Unknown read-advice policy')
    module=sys.modules.get(type(loader._pcache).__module__)
    if module is None or not callable(getattr(module,'uncache',None)):
        raise RuntimeError('Cannot locate audited vendor uncache function')
    return RuntimeReadAdvice(module,paths)


class ColdController:
    def __init__(self,mode,root,log,warmup_requests,read_advice='native'):
        if mode not in ('app-cold','page-hot','ssd-cold'):raise ValueError('Invalid mode')
        if read_advice not in ('native','preserve'):raise ValueError('Unknown read advice')
        if mode=='page-hot' and read_advice!='preserve':raise ValueError('page-hot requires preserve: native uncache would evict on read')
        self.read_advice=read_advice
        self.mode,self.root,self.log=mode,Path(root).resolve(strict=True),Path(log)
        if warmup_requests<1:raise ValueError('Warm runtime first; require at least one unmeasured request')
        self.warmup=warmup_requests;self.count=0
        if self.log.exists():raise FileExistsError(self.log)
        self.log.parent.mkdir(parents=True,exist_ok=True)

    def install(self,runner):
        old_decoder=runner.flexgen_sparse_decoder_logits
        def decoder(*args,**kwargs):
            state=_ACTIVE.get()
            if state is None or state.get('measured'):return old_decoder(*args,**kwargs)
            before=io_snapshot();result=old_decoder(*args,**kwargs);after=io_snapshot()
            state['io_first_decoder']=delta(before,after);state['measured']=True
            return result
        runner.flexgen_sparse_decoder_logits=decoder
        old=runner.greedy_flexgen_completion
        def complete(*args,**kwargs):
            self.count+=1
            if self.count<=self.warmup:return old(*args,**kwargs)
            loader=kwargs['loader'];start=time.perf_counter()
            record={'mode':self.mode,'request_index':self.count-self.warmup,
                    'task':loader._info.task,'status':'preparing','runtime_read_advice':self.read_advice}
            from contiguous_fuxian.sharedkv.io import query_hash
            record['query_hash']=query_hash(kwargs['query_token_ids'])
            try:
                paths,app=evict_application(loader,self.root)
                pages=prepare_pages(paths,self.root,self.mode)
                record.update(app);record.update(pages)
                record['setup_ms_outside_ttft']=(time.perf_counter()-start)*1000
                record['cold_condition']='payload only; model/readers/K4 index/metadata warm'
                record['physical_ssd_controller_cold_claim']=False
            except Exception as e:
                record.update(status='invalid_cold_precondition',error=str(e))
                with self.log.open('a') as f:f.write(json.dumps(record)+'\n')
                raise
            token=_ACTIVE.set(record)
            try:
                with read_advice_context(loader,paths,self.read_advice) as advice:
                    try:result=old(*args,**kwargs)
                    finally:
                        # No worker may use the scoped policy after restoration.
                        if advice is not None:
                            loader._pcache._prefetch_executor.submit(lambda:None,priority=1e12,label='v23_advice_drain').result(timeout=120)
                            torch.cuda.synchronize()
                            record['native_read_dontneed_suppressed_calls']=advice.suppressed
                record['logits_ready_ms']=float(result[1]);record['status']='completed'
                record['io_scope']='first decoder interval; process storage accounting, NOT hardware H2D/SSD bytes'
                with self.log.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                return result
            except Exception as e:
                record.update(status='request_failed',error=str(e))
                with self.log.open('a') as f:f.write(json.dumps(record)+'\n')
                raise
            finally:_ACTIVE.reset(token)
        runner.greedy_flexgen_completion=complete
