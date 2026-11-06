# 最新v2.2审核与下一轮设计

版本：utazh/pans，v2.2，fddabb7015c66c870a11810069bfbe20d5a34bbf。
依据：EXPERIMENT_REPORT_V22_ZH.md、sharedkv_v21/runtime.py、既有reader代码、flexgen_pcache.py、vendor/flexgen/my_pcache_fast.py、priority_executor.py及全局预算分配代码。GitHub结果是用户服务器的报告，不是本容器独立GPU复现。

## 观察与不能推出的结论

- 头对齐200步后，18→19局部均值RMSE由0.213450到0.197090，26→27由0.124503到0.105417；但最大请求误差分别从0.340023到0.375150、从0.198071到0.208173。不能写成所有请求都改善。
- 175条回归准确率，identity70.86%，diag19 68.00%，map19 66.86%，diag27 72.00%，map27 71.43%。没有头对齐提升分类的证据。
- 四轮平均TTFT中diag19确有相对identity的改善区间，但伴随较差质量；两组head-aligned相对identity的区间跨零。不能概括为所有配置都没有观察到任何速度差，也不能把局部最好点当质量保持加速。
- 两条profile路径的H2D约0.91ms，仍为热运行语境；不能据此证明SSD冷恢复不会有效，也不能保证冷恢复会快。
- 175条已在此前迭代观察，是开发/回归，不再是本轮新holdout。416项服务器测试不是本补丁的测试成绩。

## 确認的结构限制

现有运行时允许多个consumer引用同一source；要求source自身原生、同P8窗、leader原生、source/consumer实际selected IDs相同。它不限制为孤立两层对。

exact global block budget会产生77/78 blocks的层间差异。对于4940/16、28层、25%，总预算2163，不能要求28层均为77.25。于是朴素每四层选leader为唯一source可能不满足sameIDs。这是加载合约问题，不是模型拟合误差。本补丁先审核真实plan；将不匹配层保留原生，不暗改预算。

pilot选择[24,28)、source26、native24；消费者25和27共同校准。之后再尝试多个组的14来源目标，而不是反复只优化26→27。

## 新发现：磁盘分支主动驱逐page cache

`uncache(path)`就是posix_fadvise(DONTNEED)，`_chunk_source_readonly`等磁盘读取路径先调用它再memmap。因此，之前“提前读热文件就是page-hot控制”的方案不完整。新冷实验保留native策略的主对照，并提供明确的preserve策略，分别应用于identity/shared的page-hot与page-cold控制，避免把读时反复驱逐混成缓存冷热效应。该修正来自实际代码，而不是推测吞吐瓶颈。

## 校准改动与依据

v2/v2.2主要匹配attention局部输出、mass/分布和pre-MLP代理，且只采有限prefill位置。新方案捕获完整query及实际候选标签前缀，联合重放多层的Q/K/V、RoPE、attention、O投影、残差、MLP，匹配组末输出。固定主干参数但允许activation梯度，是组级重建，不等同于以前逐个reader增加训练步数。

这借鉴BRECQ的block reconstruction和GPTAQ的实际student输入对齐干净teacher目标；二者不是当前Qwen场景的效果证明。KV实际共享、读侧矩阵移项也不是本包声称首次提出的技术。

新贡献候选应当是“冻结主干场景下，面向一源多消费者的全序列组级读取校准，扩大可用共享范围”，而不是“冷缓存测得更快”或“多训练几步”。是否构成有效贡献需要同结构/同训练预算消融、未见数据和真实端到端验证。

## 尚未实现与实验风险

- 不处理不等选择集合；union/mask需明确新设计与公平基线。
- 不训练所有28层同时；逐组carry要重采实际student，并最终完整网络验收。
- 早层组只匹配其边界，不反传到后续全模型lm_head；last group的head KL才合法。
- 标签候选前缀用于无gold-label蒸馏，仍有task/template适配性，不能冒充跨任务泛化或长生成能力。
- 样本少时组目标仍可能过拟合；保留独立val与回归、原局部门限失败信息。
- 共享后读取算子数不减少；更多consumer增加reader成本。真正收益要看独立source减少多少恢复、I/O请求和缓存维护。
- SSD冷只认证应用payload空和OS页冷，index/readers/文件元数据仍热。设备介质类型及物理I/O需另外验证。

建议同编号reader为主线，头对齐保存为既有负/中性消融，不继续组合新的映射技巧。冷恢复先用已有reader执行，多消费者组训练单独推进；两边都成功后再合并结论。
