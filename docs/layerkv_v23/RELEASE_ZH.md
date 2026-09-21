# LayerKV v2.3 发布说明

基线：`fddabb7015c66c870a11810069bfbe20d5a34bbf`（GitHub v2.2）。
本次在服务器独立 publication 目录整理，原实验目录和原始 prism_pans 未修改。

## 代码变化

1. 新增 `sharedkv_v23`：冻结真实 decoder block，联合校准同一 source 的多个消费者。最后四层 [24,28) 为试点，26供25、27读取，24保持原生。
2. 完整 query 与所有实际候选标签续写的边界损失；只训练 A/B/bk/bv。增加相同 fit64/val64、200步、seed23 的独立层校准和 query-only 对照。
3. 增加逐请求 SSD 冷缓存协议：清空应用 payload、对目标文件 DONTNEED、mincore 全零验收；使用独立 inode 副本，不执行全局 drop_caches。冷准备不计入 TTFT。
4. 增加明确 Pcache 路径、同进程 launcher hook 恢复、现存文件逐字节验证后的持久化元数据修复；旧受保护 v2.1/v2.2 接口保持不变。
5. 增加完整500条质量、配对计时、标签 margin/翻转、运行顺序敏感性、Nsight/VizTracer、物理共享及 private decode 存储审计。
6. 保留全部失败前置检查及恢复原因。原始 GPU 实验已完成；本次发布不重新训练、不重新运行性能实验。

## 实测结论

- 完整500条：无共享343题正确，联合共享342题正确；独立层校准337题，query-only 341题。
- SSD-cold/native：2067.87→1962.81ms，下降5.08%，95%批次bootstrap区间为下降2.87%～7.35%。共10个不重叠的50条配对批次，运行顺序平衡。
- 普通缓存8轮×同32条：269.15→227.92ms。顺序影响明显；共享先运行四轮下降3.98%，区间跨零，不能宣称稳定加速。
- 唯一所选 prefix KV 降低7.12%，物理来源28→26；不是模型总显存或持久磁盘文件减少7.12%。
- Nsight冷缓存8条：H2D 135.103→130.034MB，累计时间9.390→9.167ms；主要瓶颈在主机恢复与等待，不能把约105ms的正式平均收益全部归因于PCIe拷贝。
- 准确率差的配对95%区间为[-1.2,+0.8]个百分点；旧局部门限及预设非劣验收仍未通过。
- 开发集128条联合方案下降3.125个百分点，没有扩展共享拓扑。g14/g7/g4配置是后续候选，不代表已经完成对应实测。
- 训练仅一个种子；测试500条此前已被观察。本轮结果是固定条件的历史回归，不是新的泛化或部署认证。

## 文件与复现边界

- 算法与入口：`src/contiguous_fuxian/sharedkv_v23/`、`scripts/run_sharedkv_v23.py`、`scripts/train_group_sharedkv_v23.py`。
- 固定质量与计时配置：`configs/v23_main_jobs.json`。
- 正式冷阶段在新进程恢复：`configs/v23_cold_resume_jobs.json`；普通缓存追加确认：`configs/v23_time_confirm_jobs.json`。
- `scripts/run_layerkv_v23_suite.py` 执行单一 jobs JSON；输出必须采用新的路径，避免覆盖已有结果。
- `scripts/*wait*.py` 和 `resume_v23_main.py` 是服务器编排记录，依赖已有日志/目录，不是通用安装步骤。
- `scripts/install_sharedkv_v23.py` 只用于向干净基线安装上游模块；本分支已经整合服务器适配，不要重复覆盖。
- 服务器环境：Python3.10、Torch2.10.0+cu130、Transformers4.57.6、vLLM0.19.0提供FA2；Nsight2025.5.1、VizTracer1.1.1独立采集。
- 批处理脚本与配置保留服务器实测绝对路径，必须按本机模型、数据、Pcache、GPU和解释器位置审计后调整，不能仅设置环境变量即视为完成迁移；不得移除内容hash guard。

模型、reader checkpoint、原始任务数据、prefix/payload cache、大型原始profiler及工具二进制未放入Git。
它们和完整交付包继续保留在实验服务器。这里发布代码、配置、上游补丁、四份校准报告、轻量统计、图表及验证证据。

完整报告中的 `results/`、`audit/`、`assets/` 指服务器原实验目录。Git中的对应快照位于 `docs/layerkv_v23/results/`、`audit/`、`calibration/`。
发布的 `v23_final_analysis.json` 保留全部聚合统计与有害/有益翻转UID，省略重复逐题label margin行；完整原始文件仍在服务器。
聊天记录和登录凭据没有发布。源码/发布文件的SHA256与任何内容转换见 `audit/publication_source_sha256.json`。

## 验证

原服务器完整验证：**461 passed、5 skipped、87 subtests passed**，保留一项既有NumPy只读数组警告。
8条最多16token生成中，两方案均检查到28层独立K/V存储；物理prefix alias、P8选择、数据隔离通过，561个既有文件哈希未变。
本次发布副本另运行CPU测试（CUDA隐藏），其结果见 `validation/publication_cpu_tests.txt`；没有重跑GPU实验。

发布副本实测：**441 passed、25 skipped、87 subtests passed**（CUDA隐藏）。v2.3七个接口内容hash guard通过；38个新增代码/测试文件与服务器逐字节一致。
