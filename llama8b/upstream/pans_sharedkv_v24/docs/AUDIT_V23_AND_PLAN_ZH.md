# 对已发布 v2.3 的审核与下阶段设计

## 证据与边界

依据用户提供的 EXPERIMENT_REPORT_V23_ZH.md（115行）与GitHub `v2.3` @ `3868c6b64332f89cc0b301c75b53c35630302f0e`。
报告中的500条为历史回归；它不是本轮或下一轮从未见过的测试集。

报告支持：
- source26同时服务25/27；28份来源降为26，selected prefix减少7.12%；包括reader/packed后约净减5.99%，不是模型总显存或SSD物理存储减少。
- SSD-cold/native在10批×50个互不重叠query配对中下降5.08%，区间[-7.35%,-2.87%]；1000次零驻留页检查通过。这个冷恢复延迟信号值得保留。
- 无共享343/500，联合342/500，harmful4/beneficial3，差值区间[-1.2,+0.8]pp。McNemar p=1不是非劣证明；预设1pp保守验收未过。
- dev128为99→95；不能因为已观察500只少一题，就覆盖dev停止条件。
- 独立校准dev100、联合95；group objective约0.046303 vs0.045939的差异很小，仅单seed。没有证据宣称联合校准在所有数据划分都更好。
- warm八轮整体虽降，顺序分组一边区间跨零，不能称顺序无关稳定加速。

## 实际代码发现

`sharedkv_v23/train.py::loss_for`：边界hidden重建为主，加consumer中间层重建。即使logit_weight>0，也是各个位置的全词表KL；没有直接比较完整候选标签的平均logprob顺序。
`train.py`：checkpoint按 validation_objective 选择；一次只更新当前四层组的readers；携带的前组reader被冻结，不共同纠正整个最终决策。
`capture.py`：确实捕获完整query和候选prefix路径，不应再说v23只采几个Q。缺口是没有以真实候选评分为直接目标，不是“完全没有续写数据”。
`block.py`：真实多层MLP/private KV重计算已实现，复用它，不再新增另一个相似的假block模块。

所以v24保留计算结构，新增实际score目标与全部student层的实时联合重放。它更任务条件化，不能直接推广为通用长生成蒸馏。

## 为什么平均分数差可以比hidden MSE更贴近分类

令 s_T(y)、s_S(y) 为候选平均token logprob；m_T是teacher第一、二候选的差。
把每个向量减掉自身均值后，设最大绝对误差为epsilon。若 m_T > 2*epsilon，则student仍选择teacher最优候选（纯代数充分条件）。
这并不证明teacher答案正确，也不是对新数据的非劣保证；但它对应当前runner的决策边界。
新报告逐query输出这个充分条件是否成立，不只报告平均KL。没有使用gold标签训练。

## 来源数目标

不是把28层压成1份KV，而是先做26→24→22→20，再研究16/14。
P8、同IDs、leader原生、一跳ownership保持。四层块中第一层保留；另三层以fit上的IDs等价类选中间source。这是确定的结构提案，不是最佳层选择算法。
默认前4层不共享；显式解除后才能研究14份来源。未通过val IDs合约时拒绝，不以union/减少token预算掩盖不一致。

## 冷恢复的额外诊断

完整native冷：进程read_bytes约128.866MB，而selected约70.648MB。另8条preserve对照read_bytes恰接近selected。二者样本量/读取策略不同，不能直接用它们的差异精确归因重复I/O。

代码链条提供了一个可检验解释：
`PrefixKVLayer._chunk_source_readonly` 读盘预取 -> `FlexGenLayerLoader.resolve` -> `_update_loaded_layer_score` -> CKLFU `update_score/insert` -> `TokenCacheDisk.to` 再读盘并晋升。
`TokenCacheDisk.to` 的磁盘路径调用uncache、memmap和to(device)，不复用已经恢复的请求buffer；native的DONTNEED可能使重复访问再次读盘。

v24 audit只记录两条路径调用及共同文件，未直接修改admission。不把“存在调用链”写成“全部多出来的字节已经被解释”。H2D计数、文件读取尝试、proc read_bytes是不同口径。

保留native完整结果；把冷/preserve扩到同样样本/轮次，identity/shared同策略。原有preserve从请求开始的零页状态执行，没有放弃SSD cold，只是不在请求中重复驱逐刚读过的页。这个I/O控制本身不是KV共享的新贡献。

## 验收建议

1. 原型g26：v23组边界 vs v24评分目标；相同fit/val、初始化、seed，另报告真实训练工作量。不能只说都是200步就公平。
2. 不同seed24/25/26；dev较小应增加预先划分的数据，不用重复seed冒充独立query。
3. 准入后每次只加一个4层块的共享消费者，新旧reader联合重放校准；原P8线上完整评估。
4. 核心曲线：每个G下的独立dev准确率、harmful/beneficial、实际selected净字节、冷native/preserve TTFT及真实读取。候选失败也保留。
5. 至少一个新prefix/任务完成再论一般性。当前代码score目标直接使用任务候选空间，必须做跨任务验证或每任务单独校准并报告成本。

研究主张应是“保持实际读取结构的跨层共享，在质量预算内减少独立历史记忆来源，并节省持久化恢复成本”；经典蒸馏、margin和源数枚举不是独立首次贡献。
