# 核对过的材料

审阅日期：2026-09-23；本项目主提交 11fde71c36ceecded60d560620900dec5c3a5d76。

## 用户材料
- EXPERIMENT_REPORT_V26_ZH.md：完整结果、来源图、资源、机制与统计边界；本包不修改报告或旧认证。
- scripts/v26_common.py；scripts/v26_training_worker.py；scripts/run_round4.py。
- sharedkv/runtime.py；sharedkv_v25/runtime.py；sharedkv_v26/control.py；scripts/run_evidence_v26.py。

## 原始研究与作者代码
- ContiguousKV, arXiv:2601.13631v1: https://arxiv.org/html/2601.13631v1
- KVSharer, arXiv:2410.18517v1: https://arxiv.org/html/2410.18517v1
  作者代码：https://github.com/yangyifei729/KVSharer @ 39e5c19176940f8d0089829fa8aab7992f85cb37
- xKV revised v2, arXiv:2503.18893v2: https://arxiv.org/html/2503.18893v2
  作者代码：https://github.com/abdelfattah-lab/xKV @ 05d91ecb0d698279aa4220fda5d7a5108036d692
- MiniCache, arXiv:2405.14366: https://arxiv.org/abs/2405.14366
- Cross-Layer Attention, arXiv:2405.12981: https://arxiv.org/abs/2405.12981

本包数学参考为独立实现；不是未经声明复制完整作者系统，也不宣称已有真实模型/速度结果。xKV参考的前处理、P8选择与质量-only运行边界见BASELINES_ZH.md。
