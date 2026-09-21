# v2.4 发布说明与复现边界

本次发布继承 v2.3（3868c6b64332f89cc0b301c75b53c35630302f0e），同步已完成的服务器实验代码与证据。代码来源为服务器校验过的 pans_sharedkv_v24_server_impl.zip；原实验目录和 prism_pans 不因发布而修改。

## 实现

- src/contiguous_fuxian/sharedkv_v24：拓扑规划、教师记录、全模型候选续写评分训练、恢复审计。
- faithful.py：复用在线 vLLM FA2 前向和 packed 读取变换，逐 token 续写；提供 FP32 数学 attention 反向及量化直通估计。不是重新更换在线 attention 内核。
- scripts/train_budget_sharedkv_v24.py、plan_sources_v24.py、run_sharedkv_v24.py：训练、来源规划与审计入口。
- scripts/run_v24_ladder_and_eval.py 与 run_v24_cold_jobs.py：开发集准入、冻结后评估及独立进程冷计时编排。
- tests/test_sharedkv_v24.py 与 test_sharedkv_v24_server.py：数学、模拟接口、回放梯度与运行约束测试。
- upstream/pans_sharedkv_v24 保存原输入补丁；实际执行版本位于 src/scripts/tests。不能将 upstream 原样覆盖修复后的运行版本。

实际仅执行 G26（source26 → consumer25、27）。G24/G22 等配置是未运行的后续阶段，不能作为扩层成功证据。所有 reader 保持 diagnostic-only；三个种子的新开发集非劣门限均未通过。

## 结果与复现输入

完整报告位于仓库根目录 EXPERIMENT_REPORT_V24_ZH.md。
docs/layerkv_v24/results、audit、calibration、validation 与 figures 是服务器结果的轻量副本。
报告中的 results/、audit/、assets/ 等相对路径描述原服务器布局；公开轻量副本在上述 docs 目录。
不分发模型权重、reader checkpoint、原始问题文本、teacher 记录、KV 文件、全量逐请求记录和原始 profiler。不得把缺少这些输入的空克隆视为完整可执行实验包。

需要按报告准备固定模型、prefix store、Pcache、索引、fit64/val64、dev768/final768/test500，保持原预算和查询排除规则。配置含服务器绝对路径；复现时改为自己的路径，并使用新的输出目录，保留防覆盖和数据隔离检查。用于检查旧服务器文件的历史审计脚本不是通用空仓库安装检查。

原始输入包 SHA256：cda2b01e616e72c5d2de2ec6195ab9cd32919926347fba72c55945359c9390aa。
服务器代码结果包 SHA256：acad3d04b8bb8571841f2098b696144f28a74a40aa90c77c5e3c7b4aa51b8a6b。
新文件来源哈希见 audit/publication_source_sha256.json。三个历史文件继续保留 v2.3 的既有公开版路径兼容/地址脱敏处理，未回退为服务器本机路径版本。

## 验证

服务器完整 GPU 回归记录：validation/server_all_tests.txt（510 passed、4 skipped、87 subtests passed）。
发布副本另以禁用 CUDA 的方式运行完整 tests，结果见 validation/publication_cpu_tests.txt；该结果用于验证公开副本，无新增模型训练或 GPU 实验。
v24 的 SciPy 额外依赖固定为 1.15.3，见 requirements-v24.txt。NumPy、Transformers 等沿用既有 requirements；真实回放还需要与实测相容的 torch、vLLM 和 CUDA。轻量 CPU 测试无需模型权重。

## 测量结论

质量与速度分别验收。新配方改善候选得分保真度，但不等于分类非劣已经成立。
冷恢复两种读策略分别与同策略 identity 配对；普通缓存原实验和事后复核方向相反，全部保留。Profiler 只作归因，其耗时不进入正式 TTFT。
文件读尝试、所选 payload、进程 read_bytes 和硬件 H2D 是不同口径，不互相替代。
