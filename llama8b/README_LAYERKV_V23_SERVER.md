# v2.3 服务器实验

工作目录：/home/panzihang/src/prism_layerkv
原工程 /home/panzihang/src/prism_pans 只读。所有代码、缓存副本、校准数据、结果及 profiler 均在服务器。

## 方法与固定设置

Qwen2.5-7B-Instruct，TREC prefix 4940 tokens，25% exact block budget，P8，hybrid/vLLM FA2，packed readers，metadata cache。
最后四层 [24,28)：24 原生；26 的 prefix KV 同时供 25、26、27 使用，25 和 27 各自有 A/B/bk/bv。
本层产生 Q；变换 Q 后访问同一物理 prefix K/V；变换 prefix attention 输出；与本层 private query/generated KV 联合归一化。
模型 Q/K/V/O、norm、MLP 全部冻结。实际在线运行沿用受内容哈希保护的 v2.1 后端，没有改动原工程或受保护接口。

校准来自官方 TREC 训练集：fit64、val64，另有开发集128。排除 prefix demonstrations、测试集重复文本，并验证实际 tokenizer query hash 无交集。
完整官方测试500使用原有 prefix/prompt/seed42 UID。它是历史上已观察过的回归集，不能称为全新泛化测试。
训练数据和所有文件哈希见 data/v23_splits/provenance.json、audit/v23_data_isolation.json。

## 三个冻结候选

- assets/v23_joint/trec.pt：完整冻结四层联合边界损失，包括 query 和所有实际候选标签续写分支。
- assets/v23_independent/trec.pt：相同数据/200步，各消费者使用 teacher 层输入独立重建本层输出；不传播前一个 reader 的误差。checkpoint 仍按同一验证 group objective 选取。
- assets/v23_query/trec.pt：只用 query 的联合训练，200步。
- assets/v23_query_full_validation/trec.pt：仅重新评估 query-only 候选的全部续写分支，0步、参数逐项不变；不是第四个模型。

所有候选保留 diagnostic_only/旧局部门限状态，不自动宣称质量合格。没有用 test500 调参或挑选 reader。
数据及主对照在 audit/v23_preregistered_plan.json 中固定。

## 冷缓存定义

runtime_data/v23_pcache_ssd 是独立 inode 的完整 Pcache 副本，位于 /dev/sda4 SSD。
模型、reader、K4 index、kernel、allocator 和配置 metadata 已预热。
每个计时请求前：
1. 等待 worker 和 CUDA，使用 CKLFU 的实际驱逐方法清空应用 payload。
2. 验证缓存文件归当前用户所有、regular、非硬链接；不删除文件，不执行全局 drop_caches。
3. 对真实 target prefix 文件调用 DONTNEED，再用 mincore 验证全部页为零；失败则该请求无效并中止。
4. 准备工作不计入 TTFT；原 first-logits-ready 边界保持不变。标签续写评分也不计入 TTFT。
5. /proc/self/io 仅统计 first decoder 区间，是进程存储记账，不是硬件 H2D 或 NAND 流量。
6. “SSD-cold”不保证 SSD 控制器内部缓存为空。

native 策略保留 vendor 每次读文件时的 DONTNEED。page-hot 对照必须使用 preserve，且同时跑 SSD-cold/preserve 对照；不能把 page-hot/preserve 与 cold/native 差值解释成纯页缓存效应。

完整冷测试为10个互不重叠的50条批次，两个方案每批配对，以 ABBA 顺序平衡先后次序；每个方案合计全部500条。
完整普通缓存测试：identity、joint、independent、query 各500条；另外 identity/joint 做8轮32条配对计时。
Nsight/VizTracer 另跑，不混入上述延迟。

## 对上游补丁的服务器适配

- run_sharedkv_v23.py：增加明确的 --pcache-root，并在 finally 恢复 Runtime.install / loader owner hook / runner.main，避免同进程批次污染。
- cold.py：对“文件已存在但常驻对象 file_exist=False”逐字节验证现存文件和 tensor，完全一致才补充元数据；不重新写 payload。缺失、内容不符、路径不安全均拒绝。
- cold.py：公共目录一次解析、每个文件仍逐项 lstat，复用 libc 描述；所有页驻留检查、文件身份复核不变，减少计时之外的冷准备开销。
- train.py：增加独立层校准对照和进度记录。
- 仅新增 v23 文件/脚本；既有受保护源文件、v1/v21/v22实验保留。

## 运行入口和证据

Python：/home/panzihang/venvs/vllm-stable/bin/python
设置 PYTHONDONTWRITEBYTECODE=1，OMP_NUM_THREADS=4，并确认目标 GPU 空闲。
已有结果不覆盖；复现时复制相应 jobs JSON 并给 output/stats/cold-log 换新路径。

- scripts/run_layerkv_v23_suite.py + configs/v23_capture_jobs.json：完整 query/forced suffix 采集。
- scripts/train_v23_suite.py：三种训练对照。
- scripts/run_layerkv_v23_suite.py + configs/v23_main_jobs.json：固定质量和延迟矩阵。
- scripts/profile_v23_suite.py：独立 Nsight/VizTracer。
- scripts/analyze_v23_results.py：完整配对质量/标签 margin/选择集合/延迟 bootstrap/cold 审计。
- scripts/check_v23_invariants.py：physical alias、private tail、预算、数据隔离、原文件哈希。
- scripts/plot_v23_results.py：静态实验图。

正式输出完成后见 EXPERIMENT_REPORT_V23_ZH.md、results/v23_final_analysis.json。
完整原始记录保存在 results/，profiler 原始文件在 profiles/v23/，执行命令和环境遥测在 configs/ 与 logs/。

补充：普通/冷混合长suite首个冷请求mincore未清零，结果无效保留；正式冷阶段改为独立新进程。另因首轮普通缓存baseline偏慢，增加新进程4轮确认，合计8轮全部保留，不删异常轮。


Profiler启动补充：NVTX要求显式 --diagnostic，包括identity。scripts/profile_v23_suite.py已统一补齐该标志；首次拒绝发生在模型/采集启动前，失败日志保留为*.failed_nvtx_startup，正式非profiler计时不受影响。
