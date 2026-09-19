"""Opt-in pans integration; existing runner and Pcache remain unchanged on disk.

Only the fixed-P8 ProMixed, same-selected-IDs, batch-one Qwen2 path is supported.
The original greedy completion and label-continuation scoring are retained.
DynamicCache contains PRIVATE tail only. Shared prefix lives in request state,
so label-continuation forks copy only tails and read the same immutable prefix.
"""
from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import torch
from .core import Reader, apply_rope, segmented_attention
from .io import FORMAT, BASE_COMMIT, load_safe, geometry, query_hash, validate_sources

_ACTIVE: ContextVar = ContextVar("pans_shared_prefix_request", default=None)


@dataclass
class RequestState:
    sources: list[int]
    readers: dict[int, Reader]
    backend: str
    bank: dict = field(default_factory=dict)
    scheduled: set = field(default_factory=set)
    resolve_calls: int = 0
    geometry: dict = field(default_factory=dict)

    def schedule_window(self, loader, start: int, end: int):
        # Every logical layer was configured by the original selector first.
        # Scheduling consumers here would negate the transfer reduction.
        unique = []
        for layer in range(start, end):
            source = self.sources[layer]
            if loader._selected_by_layer[layer] != loader._selected_by_layer[source]:
                raise RuntimeError("Shared consumers selected different token IDs; union mode is not implemented")
            if source not in self.scheduled:
                unique.append(source)
                self.scheduled.add(source)
        for source in unique:
            loader.schedule(source)

    def resolve(self, loader, layer, *, dtype, device):
        source = self.sources[layer]
        if source not in self.bank:
            k, v = loader.resolve(source)
            if k.ndim != 3 or k.shape != v.shape:
                raise RuntimeError("pans payload layout changed; expected [token,kv_head,dim]")
            # Convert/layout exactly ONCE per physical source, not per consumer.
            k = k.to(device=device,dtype=dtype).contiguous()
            v = v.to(device=device,dtype=dtype).contiguous()
            self.bank[source] = (k.unsqueeze(0),v.unsqueeze(0))
            loader._loaded[source] = (k,v)
            self.resolve_calls += 1
        return self.bank[source]


def decoder_logits(*, model, input_ids, cache, position_start, loader,
                   period_size, subperiod_size, impress_selection_block_size=1,
                   impress_period_prefetch_size=1, impress_selection_period_size=1,
                   impress_known_period_prefetch=False):
    state = _ACTIVE.get()
    if state is None:
        raise RuntimeError("Shared decoder called outside its request scope")
    core = model.model
    hidden = core.embed_tokens(input_ids)
    b,t,_ = hidden.shape
    if b != 1:
        raise ValueError("Shared-prefix v1 requires batch=1")
    positions = torch.arange(position_start,position_start+t,device=hidden.device)
    position_ids = positions.unsqueeze(0)
    cos,sin = core.rotary_emb(hidden,position_ids)
    first = cache.get_seq_length(0)==0
    # Imported at call time to preserve current repository implementation.
    from contiguous_fuxian.flexgen_qwen_reprefill import configure_online_layer_selection
    g = state.geometry
    for lid,layer in enumerate(core.layers):
        if first and lid % impress_selection_period_size == 0:
            chosen_period = configure_online_layer_selection(
                decoder_layer=layer,hidden_states=hidden,position_embeddings=(cos,sin),
                layer_index=lid,loader=loader,period_size=period_size,
                impress_selection_block_size=impress_selection_block_size,
                impress_selection_period_size=impress_selection_period_size)
            if chosen_period != impress_selection_period_size:
                raise RuntimeError("Adaptive selector periods are not supported")
            state.schedule_window(loader,lid,min(g["layers"],lid+chosen_period))
            loader.prefetch_selector_keys(lid+chosen_period)
        pk,pv = state.resolve(loader,lid,dtype=hidden.dtype,device=hidden.device)
        residual = hidden
        x = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(x).view(b,t,g["q_heads"],g["head_dim"])
        k = attn.k_proj(x).view(b,t,g["kv_heads"],g["head_dim"])
        v = attn.v_proj(x).view(b,t,g["kv_heads"],g["head_dim"])
        q,k = apply_rope(q,cos,sin),apply_rope(k,cos,sin)
        # Existing DynamicCache API; cache length intentionally excludes prefix.
        tk,tv = cache.update(k.transpose(1,2),v.transpose(1,2),lid,
                            {"cos":cos,"sin":sin,"cache_position":positions})
        out = segmented_attention(q,pk,pv,tk.transpose(1,2),tv.transpose(1,2),
                                  reader=state.readers.get(lid),backend=state.backend)
        hidden = residual + attn.o_proj(out.reshape(b,t,-1).contiguous())
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return model.lm_head(core.norm(hidden))


class SharedRuntime:
    def __init__(self, *, mode, backend, adapter_dir=None, stats_path=None):
        if mode not in ("identity","shared") or backend not in ("reference","flash2"):
            raise ValueError("Unsupported runtime configuration")
        self.mode,self.backend = mode,backend
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.stats_path = Path(stats_path) if stats_path else None
        self.prepared = {}
        if mode=="shared" and adapter_dir is None:
            raise ValueError("--adapters is required for shared mode")

    def _prepare(self,model,loader,period):
        task = loader._info.task
        if task in self.prepared:
            return self.prepared[task]
        g = geometry(model.config)
        if model.config.model_type != "qwen2" or getattr(model.config,"use_sliding_window",False):
            raise ValueError("Only full-attention Qwen2/Qwen2.5 is supported")
        if getattr(model.config,"rope_scaling",None) is not None:
            raise ValueError("v1 requires the unmodified fixed RoPE configuration; scaling needs additional validation")
        for layer in model.model.layers:
            if getattr(layer.self_attn,"sliding_window",None) is not None:
                raise ValueError("Sliding-window layer detected")
            if hasattr(layer.self_attn,"q_norm") or hasattr(layer.self_attn,"k_norm"):
                raise ValueError("QK normalization requires a different integration")
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        if dtype != torch.bfloat16:
            raise ValueError("The compiler models the repository BF16-compute/FP16-payload path")
        readers,banned = {},set()
        sources = list(range(g["layers"]))
        if self.mode=="shared":
            bundle = load_safe(self.adapter_dir/f"{task}.pt")
            if bundle["format"]!=FORMAT or bundle["base_commit"]!=BASE_COMMIT:
                raise ValueError("Unsupported bundle/repository version")
            checks = {"task":task,"token_hash":loader._info.token_hash,"prefix_tokens":loader.prefix_tokens,
                      "geometry":g,"period":period,"keep_ratio":float(loader.keep_ratio)}
            for key,value in checks.items():
                if bundle[key]!=value:
                    raise ValueError(f"Reader identity/configuration mismatch: {key}")
            if loader._selector_index is None or bundle["model_identity"]!=loader._selector_index.payload["model_identity"]:
                raise ValueError("Reader/model/adapter/config fingerprint mismatch")
            sources = list(bundle["sources"])
            if len(sources) != g["layers"]:
                raise ValueError("Shared source map length differs from layer count")
            readers = {int(l):Reader.from_state_dict(p).to(device=device,dtype=dtype)
                       for l,p in bundle["readers"].items()}
            if set(readers)!={l for l,s in enumerate(sources) if l!=s}:
                raise ValueError("Every shared consumer must have exactly one reader")
            banned = set(bundle["calibration_query_hashes"])
        validate_sources(sources,period=period)
        if self.backend=="flash2":
            # Small warm-up/probe before original logits-ready TTFT boundary.
            # This is not a CUDA parity certification; run GPU tests first.
            from .core import attention_with_lse
            q = torch.zeros(1,2,g["q_heads"],g["head_dim"],device=device,dtype=dtype)
            k = torch.zeros(1,3,g["kv_heads"],g["head_dim"],device=device,dtype=dtype)
            attention_with_lse(q,k,k,backend="flash2",causal=True)
            torch.cuda.synchronize()
        prepared = (g,sources,readers,banned)
        self.prepared[task] = prepared
        return prepared

    def install(self,runner):
        original_greedy = runner.greedy_flexgen_completion
        runtime_variant = runner.runtime_variant
        runner.runtime_variant = lambda **kw: runtime_variant(**kw)+f"+readside-{self.mode}-{self.backend}"
        runner.flexgen_sparse_decoder_logits = decoder_logits
        def completion(*args,**kw):
            model,loader = kw["model"],kw["loader"]
            period = kw["impress_selection_period_size"]
            if not loader.online_selection or loader.method!="impress" or not loader.impress_async_prefetch:
                raise ValueError("Use current online, async fixed-period ProMixed configuration")
            if not loader.promixed_policy or loader.promixed_policy.get("reuse_mode")!="fixed":
                raise ValueError("Only fixed reuse mode is supported")
            if loader.promixed_policy.get("adaptive_coverage",False):
                raise ValueError("Adaptive coverage is outside the v1 contract")
            if any(getattr(loader,n,False) for n in ("impress_rolling_period_prefetch","impress_priority_prefetch","impress_value_ordered_prefetch")):
                raise ValueError("Predictive/speculative schedulers must be disabled")
            if getattr(loader,"impress_period_prefetch_size",1)!=1:
                raise ValueError("Use --impress-period-prefetch-size 1")
            if getattr(loader,"_physical_to_logical",None) is not None:
                raise ValueError("Reordered physical token storage is unsupported")
            prep_started = time.perf_counter()
            g,sources,readers,banned = self._prepare(model,loader,period)
            prep_ms = (time.perf_counter()-prep_started)*1000
            digest = query_hash(kw["query_token_ids"])
            if digest in banned:
                raise ValueError("Evaluation query occurs in reader fit/validation set; leakage rejected")
            state = RequestState(sources,readers,self.backend,geometry=g)
            token = _ACTIVE.set(state)
            try:
                result = original_greedy(*args,**kw)
                if self.stats_path:
                    h,d = g["kv_heads"],g["head_dim"]
                    logical = sum(len(x) for x in loader._selected_by_layer)*2*h*d*2
                    unique = sum(len(loader._selected_by_layer[s]) for s in state.bank)*2*h*d*2
                    stats = {"query_hash":digest,"task":loader._info.task,"mode":self.mode,
                        "backend":self.backend,"sources":sources,"physical_resolve_calls":state.resolve_calls,
                        "logical_selected_payload_bytes_fp16":logical,
                        "unique_selected_payload_bytes_fp16":unique,
                        "request_prefix_buffer_bytes":sum(x.numel()*x.element_size() for kv in state.bank.values() for x in kv),
                        "resident_reader_bytes":sum(x.numel()*x.element_size() for r in readers.values() for x in (r.A,r.B,r.bk,r.bv)),
                        "preparation_ms_outside_logits_ttft":prep_ms,
                        "logits_ready_ms":result[1] if isinstance(result[1],(float,int)) else None,
                        "note":"Selected-byte counts are not hardware H2D bytes. Use native loader source statistics/Nsight for transfers."}
                    self.stats_path.parent.mkdir(parents=True,exist_ok=True)
                    with self.stats_path.open("a") as f:
                        f.write(json.dumps(stats)+"\n")
                return result
            finally:
                _ACTIVE.reset(token)
        runner.greedy_flexgen_completion = completion
