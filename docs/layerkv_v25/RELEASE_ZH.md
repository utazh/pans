# v2.5 发布说明与复现边界

本发布继承 v2.4（a6be130c921024a4cd358ccf5841e4a46f547ae2）。源目录为服务器 /home/panzihang/src/prism_layerkv；发布在独立 publish_v25 副本完成，不修改原实验代码、已有checkpoint或 prism_pans。

## 方法与已执行范围

- src/contiguous_fuxian/sharedkv_v25：共享/本层支持集分割、原生保留策略、双分支attention、评分回放、训练和接口哈希检查。
- scripts/run_native_prefix_v25.py、train_native_prefix_v25.py、plan_exploration_v25.py：运行、校准与拓扑规划入口。
- scripts/run_v25_experiment_pipeline.py、run_v25_timing.py：完整网格、固定三种子、冻结后留出评估以及顺序配对计时。
- tests/test_sharedkv_v25.py：数学、支持集、原生保留区、预算与运行约束验证。
- docs/sharedkv_v25_package 与 upstream/pans_sharedkv_v25 保留输入方案和原始补丁；实际执行版本位于 src/scripts/tests。
- 运行 G26/24/22/20/16/14 × R0/R4；选定G16后，两种R均补种子25/26。所有checkpoint仍为诊断候选，严格最终非劣验收均未通过。

R4不是增加预算，也不是新增滑动窗口：只从已经选中的物理块保留本层最后4块，短末块使实际长度通常为60 token。剩余prefix读取共享源的视图；本层小块与私有query/tail合并参与两分支联合归一化。28层当前query/生成KV独立，主干冻结。

## 结果与轻量发布布局

根目录 EXPERIMENT_REPORT_V25_ZH.md 是完整服务器报告，字节与交付记录一致。
docs/layerkv_v25/results、audit、calibration、validation、figures 是轻量证据副本；根目录figures另保留报告图片链接。报告中 results/、audit/、assets/、profiles/ 等路径描述原服务器布局。

不分发模型权重、reader checkpoint、原始问题文本/teacher记录、KV文件、全量逐请求结果、原始Nsight/VizTracer和85MB服务器交付ZIP。这些输入保留在服务器，空克隆不能直接复现实模型实验。所选缓冲区减少不代表持久化KV已被删除。

服务器交付ZIP SHA256：61cbad4e1d2bd8bb31fd7caa371ea1a63a9d039cce35ecda878aef28e17475ae。
逐文件来源哈希见 audit/publication_source_sha256.json；发布验证见 audit/publication_validation.json。历史审计中的绝对路径和manifest描述原服务器，不应当成空克隆的检查目标。

## 依赖与验证

沿用 requirements.txt、requirements-dev.txt、requirements-v24.txt。真实模型回放需要与实测相容的 PyTorch/vLLM/CUDA，软件版本见 audit/v25_software_environment.json。

在已有合适PyTorch环境中：

```bash
python -m pip install -r requirements-dev.txt -r requirements-v24.txt
python -m pip install --no-deps -e .
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests -p no:cacheprovider
```

服务器完整GPU验证：validation/server_all_tests.txt（547 passed、4 skipped、87 subtests passed）。
公开副本另外禁用CUDA运行CPU测试，见 validation/publication_cpu_tests.txt；它不替代GPU/真实模型证据。
v25接口guard保持启用，发布前对继承的11个基础文件核对固定Git blob哈希，不放宽检查。

## 复现实模型实验

准备同一Qwen2.5-7B模型、prefix_store、chunk16 Pcache、K4索引、fit64/val64教师记录、开发768、新留出1536及历史test500。维护已有文本和token-hash排除规则，不重新利用留出集调参。先按 configs/v25_explore 规划拓扑，再用训练/推理入口；完整编排参数在 configs/v25*_command.json 和 configs/v25*_jobs.json。

编排脚本保留服务器路径，用于追溯实际实验；迁移时应修改路径并使用全新的输出目录及自有缓存副本。不要在空克隆中直接运行历史审计/归档脚本。冷缓存清理只针对本实验拥有的目标prefix，不清理全系统缓存或其他作业。

主计时seed24固定，每种主冷场景4×16题、普通缓存4×32题；不能称作完整500条计时。704次正式冷认证覆盖应用payload及OS文件页。Profiler各8题，与无探针计时分开。硬件H2D、进程read_bytes、文件请求字节和selected KV缓冲区属于不同口径。

准确率和速度分别评估。R4在新留出三个固定种子上表现接近基线，但保守三种子非劣未通过；不宣称已经证明质量保持或跨任务稳定收益。

## Git文本换行

质量表CSV沿用Git标准文本规则，服务器CRLF换行在Git对象中规范化为LF；表格数据不变。来源SHA清单记录复制时的服务器原始字节，换行说明见publication_validation.json。代码、完整Markdown报告和图表没有这一差异。
