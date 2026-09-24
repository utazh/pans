# v2.7 审核与最小改动决定

审核基线：681fe495028598e9d51f2f57220ecb96ed0e5f35。证据为用户完整 v27 报告及 GitHub 文件，不是在本环境复跑 Qwen/SSD。

## 保留的结果和边界

TREC500的frozen-R0为76.8%，R4为73.0%，同后端identity68.6%。这500条已被多轮观察。R0与proxy总正确数一致但不是逐题/逐位相同：495条预测一致，双方分别有两条相对正确性改善。冻结R0不需要新增训练，但权重来源仍是R4训练。

三块×32条冷恢复中，R0相对identity约40.65%更快，相对R4均值仅0.79%改善且块方向不一致。当前应主张路径更简洁、少恢复对象和当前回归集点估计改善，不主张稳定的额外加速。

本包重新运行每格I/R4/R0，不把旧日期的identity计时与新运行结果混比；输出来自当前scorer。

## 源码中确认的要点

1. `frozen_control.make_frozen_r0`只改policy.blocks=0，移出旧质量report，保留来源图、A/B/bias及查询禁用名单。本包复用它并加逐张量精确核对，不另写一个训练器。
2. `scripts/v27_search_kvsharer.py`才是本轮作者距离/remap适配；旧`baseline_math.dissimilar_edges`仍不是这个公式。本包参数化实际server脚本，K/V分别求向量距离后再平均，不回退旧primitive。
3. `v27_bf16_sequence.py`从FP64参考导出BF16，而旧FP32来自随机SVD。两条线不能当成量化单变量。本包统一FP64 master和U/SVh尺度，再导出两种精度。
4. `baseline_quality.QualityMemory`先执行原native loader，再替换attention输入。因此其原TTFT不代表压缩source/因子的恢复速度。本包继续强制quality-only。
5. FP32全秩tensor误差很小，不代表真实模型已复现；报告中最大标签分差0.499757超过0.05。FP64全秩才达到真实bridge的0差。本包每格重新运行FP64全秩/直接identity的完整候选数值门限，不放宽门限。
6. `prepare_subj_canonical_v26.py`明确生成canonical_full_eval，主评估1003条。模板采用这个文件而非canonical_eval子集；每个SUBJ reader和teacher用canonical产物。
7. 同任务不同预算的来源图可能变化，同G不等于同层距/共享拓扑；本轮保留各格真实图，不为凑统一图改变整数预算。

## 改动范围

新增实验编排、同源FP64分解导出、参数化现有KVSharer搜索、资源参考账、结果提取和文档。推理仍调用v26/v25/v23入口，训练不运行。未实现外部SSD serving engine、xKV-SR landmark selector、14B/多卡。

## 论文中可说与不可说

可以：冻结主干、离线训练读取器；通过读侧变换跨层共享已命中的prefix物理记忆；frozen-R0在当前TREC回归上不需要原生小块；恢复路径在指定冷协议下有实测优势。

不能：R0训练免费；R0普遍优于R4；真实原生KV对所有任务都无用；KVSharer/xKV官方模型或系统速度复现完成；15/16旧质量下降已被单个TREC点推翻；全部模型显存/SSD磁盘占用下降42.72%。

## 下一轮停止线

先完成冻结16格及外部质量比较，不再改reader。若R0跨格并未改善，也保留这个结果，将主张限定为质量—资源折中；不按测试分数给每格挑R0/R4的最大值。需要新的最终确认时，另冻结数据与操作点，不重复认证旧历史集。
