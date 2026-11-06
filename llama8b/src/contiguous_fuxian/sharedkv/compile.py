"""Offline reader compiler for fixed-prefix calibration traces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .core import Reader, attention_with_lse, query_transform, key_offset, value_transform
from .fit import fit_key_reader, affine_ridge, fit_value_from_attention
from .io import FORMAT, TRACE_FORMAT, BASE_COMMIT, load_safe, save_new, read_prefix, validate_sources


def _traces(root, task):
    paths = sorted((Path(root)/task).glob("*.pt"))
    if not paths:
        raise ValueError(f"No traces under {root}/{task}")
    rows = [load_safe(p) for p in paths]
    for r in rows:
        if r["format"] != TRACE_FORMAT or r["task"] != task:
            raise ValueError("Wrong trace format/task")
    return rows


def _features(rows, source, target, ks, vs, kt, vt, reader):
    result = []
    for record in rows:
        data = record["layers"][target]
        ids = data["selected_tokens"]
        if not torch.equal(ids, record["layers"][source]["selected_tokens"]):
            raise ValueError("v1 requires equal selected token IDs within a shared pair")
        q = data["q"].float().unsqueeze(0)
        sp, lp = attention_with_lse(query_transform(q, reader), ks[ids].unsqueeze(0), vs[ids].unsqueeze(0))
        lp = lp + key_offset(q, reader, q.shape[-1]**-0.5)
        tp, lt = attention_with_lse(q, kt[ids].unsqueeze(0), vt[ids].unsqueeze(0))
        tail_lse = data["tail_lse"].float().transpose(0, 1).unsqueeze(0)
        bs = torch.sigmoid(lp-tail_lse).transpose(1, 2)[0]
        bt = torch.sigmoid(lt-tail_lse).transpose(1, 2)[0]
        result.append({"us":sp[0], "ut":tp[0], "uv":data["tail_output"].float(),
                       "bs":bs, "bt":bt, "hash":record["query_hash"]})
    return result


def _validation(features, reader):
    error = signal = 0.0
    mass_errors = []
    request_errors = []
    for x in features:
        corrected = value_transform(x["us"].unsqueeze(0), reader)[0]
        actual = x["bs"][..., None]*corrected + (1-x["bs"])[..., None]*x["uv"]
        target = x["bt"][..., None]*x["ut"] + (1-x["bt"])[..., None]*x["uv"]
        e = (actual-target).double().square().sum().item()
        s = target.double().square().sum().item()
        error += e
        signal += s
        request_errors.append((e/max(s,1e-30))**0.5)
        mass_errors.append((x["bs"]-x["bt"]).abs().flatten())
    mass = torch.cat(mass_errors)
    return {"output_relative_rmse": (error/max(signal,1e-30))**0.5,
            "output_request_max_relative_rmse": max(request_errors),
            "prefix_mass_mae": mass.mean().item(),
            "prefix_mass_p95_absolute_error": torch.quantile(mass,0.95).item(),
            "validation_requests":len(features)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-root", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--fit-traces", required=True)
    p.add_argument("--validation-traces", required=True)
    p.add_argument("--pairs", required=True, help="source:consumer, e.g. 2:3,4:5; disjoint pairs")
    p.add_argument("--key-mode", choices=["dense","rope","rope_rr","direct"], default="rope_rr")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--key-fit-tokens", type=int, default=8192)
    p.add_argument("--max-output-error", type=float, default=0.03)
    p.add_argument("--max-request-output-error", type=float, default=0.08)
    p.add_argument("--max-mass-p95", type=float, default=0.03)
    p.add_argument("--output", required=True)
    p.add_argument("--diagnostic-output", help="Export rejected candidates separately for explicitly diagnostic runs")
    a = p.parse_args(argv)
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    if a.key_fit_tokens <= 0 or min(a.max_output_error,a.max_request_output_error,a.max_mass_p95) <= 0:
        raise ValueError("Positive token count and validation thresholds required")
    train, valid = _traces(a.fit_traces,a.task), _traces(a.validation_traces,a.task)
    header = train[0]
    for r in train+valid:
        for field in ("task","token_hash","prefix_tokens","geometry","model_identity","period","keep_ratio"):
            if r[field] != header[field]:
                raise ValueError(f"Mixed calibration identities/configurations: {field}")
    th, vh = {r["query_hash"] for r in train}, {r["query_hash"] for r in valid}
    if th & vh:
        raise ValueError("Fit/validation query overlap; split by whole request")
    if len(th) != len(train) or len(vh) != len(valid):
        raise ValueError("Duplicate calibration requests")
    g = header["geometry"]
    sources = list(range(g["layers"]))
    candidate_sources = sources.copy()
    pairs, used = [], set()
    for pair in a.pairs.split(","):
        source,target = map(int,pair.split(":"))
        if source == target or source in used or target in used:
            raise ValueError("v1 accepts disjoint two-layer pairs only")
        if not (0<=source<g["layers"] and 0<=target<g["layers"]):
            raise ValueError("Invalid pair layer index")
        used.update((source,target)); pairs.append((source,target))
        candidate_sources[target] = source
    validate_sources(candidate_sources,period=header["period"])
    readers, reports = {}, {}
    candidate_readers = {}
    torch.set_num_threads(min(torch.get_num_threads(),4))
    with torch.no_grad():
        for source,target in pairs:
            ks, ms = read_prefix(a.store_root,a.task,source,"key")
            vs, _ = read_prefix(a.store_root,a.task,source,"value")
            kt, mt = read_prefix(a.store_root,a.task,target,"key")
            vt, _ = read_prefix(a.store_root,a.task,target,"value")
            if ms["token_hash"] != header["token_hash"] or mt["token_hash"] != header["token_hash"]:
                raise ValueError("Prefix content identity differs from the calibration trace")
            n,h,d = ks.shape
            ids = torch.linspace(0,n-1,min(n,a.key_fit_tokens)).round().long().unique()
            q = torch.cat([r["layers"][target]["q"].float() for r in train],dim=0)
            groups = g["q_heads"]//h
            qr = q.reshape(-1,h,groups,d)
            matrices, offsets, key_reports = [], [], []
            for head in range(h):
                if a.key_mode == "direct":
                    ak,bk,rep = torch.eye(d),torch.zeros(d),{"key_mode":"direct"}
                else:
                    ak,bk,rep = fit_key_reader(ks[ids,head],kt[ids,head],qr[:,head].reshape(-1,d),
                        rank=min(a.rank,d),ridge=a.ridge,mode=a.key_mode)
                matrices.append(ak);offsets.append(bk);key_reports.append(rep)
            reader = Reader(torch.stack(matrices),torch.eye(d).expand(h,d,d).clone(),
                            torch.stack(offsets),torch.zeros(h,d))
            if a.key_mode != "direct":
                fs = _features(train,source,target,ks,vs,kt,vt,reader)
                joined = {name:torch.cat([x[name] for x in fs],dim=0) for name in ("us","ut","uv","bs","bt")}
                b_list,bias_list = [],[]
                for head in range(h):
                    lo,hi = head*groups,(head+1)*groups
                    b0,c0 = affine_ridge(vs[ids,head],vt[ids,head],ridge=a.ridge)
                    prior = torch.cat((b0,c0.unsqueeze(0)))
                    args = [joined[name][:,lo:hi].reshape(-1,d) for name in ("us","ut","uv")]
                    args += [joined[name][:,lo:hi].reshape(-1) for name in ("bs","bt")]
                    b1,c1 = fit_value_from_attention(*args,ridge=a.ridge,prior=prior)
                    b_list.append(b1);bias_list.append(c1)
                reader = Reader(reader.A,torch.stack(b_list),reader.bk,torch.stack(bias_list))
            reader.validate()
            # Validate the same BF16-quantized coefficients used online (reference
            # attention still accumulates in FP32, so real GPU parity is separate).
            reader = reader.to(device="cpu",dtype=torch.bfloat16).to(device="cpu",dtype=torch.float32)
            report = _validation(_features(valid,source,target,ks,vs,kt,vt,reader),reader)
            accept = (report["output_relative_rmse"]<=a.max_output_error and
                report["output_request_max_relative_rmse"]<=a.max_request_output_error and
                report["prefix_mass_p95_absolute_error"]<=a.max_mass_p95)
            report.update({"source":source,"consumer":target,"accepted_local_only":accept,
                           "key_fit":key_reports,
                           "warning":"Local teacher-query validation is NOT end-to-end quality validation."})
            reports[str(target)] = report
            candidate_readers[str(target)] = reader.state_dict()
            if accept:
                sources[target]=source
                readers[str(target)]=reader.state_dict()
            print(json.dumps({"layer":target,**report},ensure_ascii=False),flush=True)
    bundle = {"format":FORMAT,"base_commit":BASE_COMMIT,"task":a.task,"sources":sources,
        "readers":readers,"token_hash":header["token_hash"],"prefix_tokens":header["prefix_tokens"],
        "geometry":g,"model_identity":header["model_identity"],"period":header["period"],
        "keep_ratio":header["keep_ratio"],"calibration_query_hashes":sorted(th|vh),
        "reports":reports,"compiler_args":vars(a),
        "scope":"post-RoPE, fixed-position, fixed-prefix; no backbone training; dense exported readers",
        "accepted_consumers":len(readers),"end_to_end_validated":False}
    validate_sources(sources,period=header["period"])
    save_new(a.output,bundle)
    Path(a.output).with_suffix(".report.json").write_text(json.dumps({k:v for k,v in bundle.items() if k!="readers"},indent=2,ensure_ascii=False))
    if a.diagnostic_output:
        diagnostic = {**bundle, "sources":candidate_sources, "readers":candidate_readers,
                      "diagnostic_only":True, "active_consumers":len(candidate_readers),
                      "warning":"Includes locally rejected readers; research ablation only."}
        save_new(a.diagnostic_output,diagnostic)
    print(f"Accepted {len(readers)} of {len(pairs)} candidate pairs. No quality/TTFT claim is implied.")


if __name__ == "__main__":
    main()
