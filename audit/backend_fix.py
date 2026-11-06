from pathlib import Path
r=Path('/home/panzihang/src/prism_layerkv')
p=r/'src/contiguous_fuxian/sharedkv/core.py';s=p.read_text().replace('cuq, cuk, tq, nk, causal=causal','max_seqlen_q=tq, cu_seqlens_q=cuq, max_seqlen_k=nk, cu_seqlens_k=cuk, causal=causal');p.write_text(s)
p=r/'tests/test_sharedkv.py';s=p.read_text();start=s.index('@pytest.mark.skipif(not torch.cuda.is_available()');end=s.index('\n\ndef test_compiler',start);new=s[start:end].replace('or importlib.util.find_spec("flash_attn") is None','').replace('test_flash2_segmented','test_vllm2_segmented').replace('backend="flash2"','backend="vllm2"');p.write_text(s+'\n\n'+new)
