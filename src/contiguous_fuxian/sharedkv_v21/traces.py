"""Paired teacher/student traces. All collection is OUTSIDE performance evidence."""
from __future__ import annotations
from pathlib import Path
import torch
from contiguous_fuxian.sharedkv.io import load_safe, save_new
TRACE_FORMAT = "pans-paired-reader-trace-v2"
IDENTITY_FIELDS = ("task","token_hash","prefix_tokens","geometry","model_identity","period","keep_ratio")


def load_traces(root, task):
    paths = sorted((Path(root)/task).glob("*.pt"))
    if not paths:
        raise ValueError(f"No request traces: {root}/{task}")
    out = {}
    for p in paths:
        r = load_safe(p)
        if r.get("format") != TRACE_FORMAT or r.get("task") != task:
            raise ValueError(f"Wrong trace identity: {p}")
        h = r["query_hash"]
        if h in out:
            raise ValueError("Duplicate request hash")
        out[h] = r
    head = next(iter(out.values()))
    for r in out.values():
        for f in IDENTITY_FIELDS+("rollout_sources","rollout_reader_fingerprint"):
            if r[f] != head[f]:
                raise ValueError(f"Mixed trace run identities: {f}")
    return out


def paired(teacher, student, target, *, require_same_ids=True):
    if set(teacher) != set(student):
        raise ValueError("Teacher/student request sets differ; no silent dropping")
    rows = []
    for h in sorted(teacher):
        t,s = teacher[h],student[h]
        for f in IDENTITY_FIELDS:
            if t[f] != s[f]:
                raise ValueError(f"Paired identity mismatch: {f}")
        if target not in t["layers"] or target not in s["layers"]:
            raise ValueError("Requested layer absent from trace")
        a,b = t["layers"][target],s["layers"][target]
        if not torch.equal(a["query_positions"],b["query_positions"]):
            raise ValueError("Teacher/student query positions differ")
        if require_same_ids and not torch.equal(a["selected_tokens"],b["selected_tokens"]):
            raise ValueError("Selected IDs differ: use fixed-plan calibration; online drift is a separate audit")
        rows.append((t,s))
    return rows


def assert_splits(fit, valid):
    if set(fit)&set(valid):
        raise ValueError("Fit/validation request leakage")
    if not fit or not valid:
        raise ValueError("Both nonempty splits are required")


def fingerprint(readers):
    import hashlib
    h = hashlib.sha256()
    for lid in sorted(readers):
        h.update(str(lid).encode())
        for key in ("A","B","bk","bv"):
            x = getattr(readers[lid],key).detach().cpu().contiguous()
            h.update(str((key,tuple(x.shape),x.dtype)).encode())
            h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class TraceSink:
    def __init__(self, root, task, digest, header, *, layers, max_samples=8):
        self.path = Path(root)/task/f"{digest}.pt"
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.header = header
        self.layers = set(layers)
        self.max_samples = max_samples
        self.records = {}
        self.root = Path(root)
        self.skip = self.path.exists()

    def record(self, lid, *, q,tk,tv,hidden_in,pre_mlp,hidden_out,out,selected,layer):
        if self.skip or lid not in self.layers or lid in self.records:
            return
        from contiguous_fuxian.sharedkv.core import attention_with_lse
        t = q.shape[1]
        positions = (torch.tensor([t-1],device=q.device) if self.max_samples == 1 else
                     torch.linspace(0,t-1,min(t,self.max_samples),device=q.device).round().long().unique())
        qs = q.index_select(1,positions)
        with torch.no_grad():
            u,z = attention_with_lse(qs,tk,tv,causal=True,q_positions=positions,backend="reference")
        def sample(x): return x.index_select(1,positions)[0].detach().cpu()
        self.records[lid] = {"q":qs[0].detach().cpu(),"tail_output":u[0].detach().cpu(),
            "tail_lse":z[0].transpose(0,1).detach().cpu(),"query_positions":positions.cpu(),
            "selected_tokens":torch.tensor(selected,dtype=torch.long),
            "hidden_in":sample(hidden_in),"pre_mlp":sample(pre_mlp),
            "hidden_out":sample(hidden_out),"attention_output":sample(out)}
        wpath = self.root/"weights"/f"layer_{lid:02d}.pt"
        if not wpath.exists():
            save_new(wpath,{"o_proj":layer.self_attn.o_proj.weight.detach().cpu(),
                            "model_identity":self.header["model_identity"]})

    def save(self):
        if self.skip: return
        if set(self.records) != self.layers:
            raise ValueError("Incomplete requested layer traces")
        save_new(self.path,{"format":TRACE_FORMAT,**self.header,"layers":self.records,
                            "warning":"Diagnostic collection; never use its timings as TTFT"})
