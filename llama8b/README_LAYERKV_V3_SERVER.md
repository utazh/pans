# v3 服务器实验执行说明（对应 v2.8 实现）
所有实施、参考文件与结果均位于 /home/panzihang/src/prism_layerkv。原 prism_pans 未修改。
输入包 pans_sharedkv_v28.zip SHA256: 48b8374ac1607f1243ddc6e5b5ebb72c8ea6e26fd93f2f54e65b9afd2292369f。
基于 v2.7 接口七项 git blob 校验。新增安装18文件。初始完整回归708通过、4跳过、87子测试通过；无损参考编码新增验证后53项v28测试通过。

## 冻结协议
configs/v28_spec.json 与 studies/v28_frozen/manifest.json：四任务、四预算，每格已有R4成对导出R0，无新增训练。
完整人数 SST-2=872、SUBJ=1003、TREC=500、RTE=277；SUBJ canonical_full_eval。
TREC25 teacher validation使用 data/v24_splits/val。
48个主质量臂，最多112个外部质量臂。
每格冷热分别6块×16个不同query×3臂，六种次序各一次。
主正式计时为GPU2串行、SSD-cold/preserve+reuse-prefetch，另测普通缓存。
模型、reader保持预热，逐正式请求认证应用payload为空且目标文件页不驻留。

## 执行资源与最小服务器修正
GPU2主质量、GPU3外部搜索/质量、CPU因子准备可以重叠；这些时间不作为正式性能结果。
configs/v28_external_manifest.json仅覆盖GPU3与独立缓存路径，其他输入/数据/数值门限不变；外部作业键与主质量不相交。
搜索使用子进程，结束后释放CUDA上下文，避免搜索模型缓存占用后续评估显存。
最终计时依赖所有质量和准备完成的标记，不与本轮其他GPU作业重叠。每个计时作业保存整机负载和GPU快照。
为避免磁盘不足，仅将新 direct-map 外部质量参考的checkpoint ZIP记录无损DEFLATE编码；读取后逐张量dtype和值完全一致，理论字节仍按原tensor计算，不是方法贡献或部署I/O优化。
初次安装因PYTHONPATH命中旧包失败，改用隔离路径安装；未覆盖旧文件。
第一外部协调进程在搜索产物写出前主动停止，改为每格搜索子进程；旧日志与停止理由保留。
未修改旧attention、selector、Pcache、v27数值接入模块。

## 进程与结果
scripts/v28_main_sequence.py
scripts/v28_factors_sequence.py
scripts/v28_external_sequence_attempt2.py
scripts/v28_finish_sequence.py
logs/下同名日志，audit/下PID、阶段状态。
当前已完成质量与部分计时的发布快照：docs/layerkv_v3/results/{quality.csv,timing.csv,external_theory_only.csv,analysis.json}；报告 EXPERIMENT_REPORT_V3_ZH.md。
原 results/v28_summary 是完整任务结束后的计划输出，本次尚未生成。
失败会保留日志并中止依赖任务，不伪造完成；外部质量桥不进入SSD速度表。

完整质量已完成，计时因严格冷态认证失败而停止。服务器编排脚本仅作原运行记录；复现需在独立目录准备输入及资源配置，不能直接覆盖既有结果。不要直接运行 v28_write_report.py 覆盖当前部分结果报告；其完整任务前置条件尚未满足。已知 v28_final_results_audit.py 的 q+1 假设不适用于全部任务，本轮实际采用报告中按标签长度的只读审计口径，见审核 JSON。
