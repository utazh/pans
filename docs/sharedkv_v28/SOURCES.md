# 本轮核对的源材料

## 用户版本

- pans v2.7: https://github.com/utazh/pans/tree/681fe495028598e9d51f2f57220ecb96ed0e5f35
- `EXPERIMENT_REPORT_V27_ZH.md`：当前回归、数值敏感性、外部适配、冷计时边界。
- `scripts/v27_search_kvsharer.py`：本轮实际作者距离/remap搜索。
- `scripts/v27_bf16_sequence.py` / `v27_compile_full64.py`：FP64参考及BF16导出。
- `scripts/run_baseline_quality_v27.py` / `sharedkv_v27/baseline_quality.py`：质量-only接入。
- `sharedkv_v27/frozen_control.py`：冻结R4→R0导出。
- `scripts/prepare_subj_canonical_v26.py`：canonical完整划分路径和修正说明。

## 作者原始资料

- KVSharer paper: https://arxiv.org/abs/2410.18517
- KVSharer author snapshot: https://github.com/yangyifei729/KVSharer/tree/39e5c19176940f8d0089829fa8aab7992f85cb37
- xKV paper: https://arxiv.org/abs/2503.18893
- xKV author snapshot: https://github.com/abdelfattah-lab/xKV/tree/05d91ecb0d698279aa4220fda5d7a5108036d692

原项目已有来源快照及SHA，本包没有重新分发这些完整作者系统，也不把前缀适配叫成官方速度复现。
