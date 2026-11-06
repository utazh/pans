# 参考方法和代码（不是本方案效果证明）

## 当前实验源码
- https://github.com/utazh/pans/tree/59312f2bd2504079fed03b53f283d8ade931fee2
- EXPERIMENT_REPORT_ZH.md、README_LAYERKV.md、docs/layerkv/results/comparison.csv
- src/contiguous_fuxian/sharedkv/{runtime,core,selection,compile,fit}.py
- scripts/{run_sharedkv,analyze_layerkv}.py

## 非对称校准
GPTAQ（早期名称GPTQv2）：https://arxiv.org/abs/2504.02692
作者代码：https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ
重点fake_quant/gptaq_utils.py：以受到上游扰动的输入匹配原全精度输出。借鉴目标，不复制其量化更新式，不能据此保证KV共享质量。

ReCalKV：https://arxiv.org/html/2505.24357v3
作者代码：https://github.com/XIANGLONGYAN/ReCalKV
重点palu/model/modules/svd_linear.py、palu/decomposition.py：离线Value校准、K/V非对称处理。我们只对共享prefix分支应用B，不能简单替换整层o_proj。

## 层间共享的强相关基线
xKV：https://arxiv.org/html/2503.18893v2
作者代码：https://github.com/abdelfattah-lab/xKV
共享子空间及选择性重建：单source线性读取能力不足时的对照，不将共同子空间换个名字声称首次。

KVSharer：https://arxiv.org/abs/2410.18517
直接KV跨层共享已有先例；本研究必须证明读侧校准、实际传输及固定prefix场景的增量价值。

## 分段attention与执行
FlashInfer论文：https://arxiv.org/abs/2501.01005
官方说明：https://docs.flashinfer.ai/tutorials/recursive_attention.html
官方API：https://docs.flashinfer.ai/api/cascade.html
代码：https://github.com/flashinfer-ai/flashinfer
学习attention状态/LSE合并与可组合布局。跨请求prefix共享不是本方案的跨层共享；不能把FlashInfer的论文加速比转用到RTX3090的本方案。
本补丁保留v1的vllm.vllm_flash_attn.flash_attn_varlen_func(...fa_version=2)接口，不强制安装新FlashInfer。

## 质量覆盖扩展
RULER：https://arxiv.org/abs/2404.06654 （可控长度/复杂度的长上下文任务）
LongBench：https://arxiv.org/abs/2308.14508 （自然长文本多任务）
要把这些数据适配为真正固定prefix、多query的协议，披露适配差异，不能伪装成原版官方分数。
