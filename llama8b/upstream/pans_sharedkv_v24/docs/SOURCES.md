# 核对过的依据和实现入口

## 用户工程与实测（首要依据）

- 分支： https://github.com/utazh/pans/tree/v2.3
- 锚定提交： `3868c6b64332f89cc0b301c75b53c35630302f0e`
- 报告：用户当前上传的 `EXPERIMENT_REPORT_V23_ZH.md`
- `sharedkv_v23/train.py`、`block.py`、`capture.py`：组边界目标、冻结层重放和实际标签prefix采集。
- `flexgen_qwen_reprefill.py::greedy_flexgen_completion`：实际候选token逐步得分。
- `flexgen_pcache.py::resolve` 与 `_update_loaded_layer_score`；`my_pcache_fast.py::TokenCacheDisk.to`：消费后的评分/晋升与磁盘读取路径。

本次通过GitHub连接器实际读取；本地直接clone因网络限制失败，未把旧main当作当前分支。测试底座使用之前交付包中与已读取源文件内容hash一致的数学模块。完整服务器checkout和GPU回归没有在本环境运行。

## 研究参考（不是本方法效果证明）

1. Hinton et al., Distilling the Knowledge in a Neural Network.
   https://arxiv.org/abs/1503.02531
   借鉴输出分布蒸馏。v24的平均候选评分KL是任务条件化目标，不是该论文直接给出的跨层KV算法。

2. Li et al., BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction.
   https://arxiv.org/abs/2102.05426
   https://github.com/yhhhli/BRECQ/blob/main/quant/block_recon.py
   解释v23组重建的来源；本轮不声称再发明block reconstruction。

3. Li et al., GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration.
   https://arxiv.org/abs/2504.02692
   https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ
   使用实际受扰动student输入匹配干净teacher，而不把teacher输入冒充student。它是量化工作，不保证KV共享成立。

4. Muralidharan et al., Compact Language Models via Pruning and Knowledge Distillation.
   https://arxiv.org/abs/2407.14679
   https://github.com/NVlabs/Minitron
   借鉴逐步压缩/蒸馏及比较输出目标的实验思路。不能把其训练成本和模型结果外推到我们的冻结reader训练。

5. Chang et al., xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction.
   https://arxiv.org/abs/2503.18893
   https://github.com/abdelfattah-lab/xKV
   跨层共同子空间及选择性重建，是后续强相关基线；本轮未声称已经复现。

## 系统/数值

- PyTorch 2.10 checkpoint（use_reentrant=False）：
  https://docs.pytorch.org/docs/2.10/checkpoint.html
  重算换激活内存，反向调用必须保持一致；本包两遍分支反传另做数值与梯度对照。
- Linux posix_fadvise：
  https://man7.org/linux/man-pages/man2/posix_fadvise.2.html
  DONTNEED是建议，不保证成功；完整页、脏页条件重要。
- Linux /proc/pid/io：
  https://man7.org/linux/man-pages/man5/proc_pid_io.5.html
  read_bytes是存储层记账，不是PCIe传输/NAND读数。

源代码接口截至本次核对。文件引用与web研究是两类证据；不要用后者替代用户实际实验数据。
