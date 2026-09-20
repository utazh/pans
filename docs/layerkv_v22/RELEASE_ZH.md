# LayerKV v2.2 发布说明

基线为 GitHub v2：b48c45d779ecde8ae960a6e4837d8b6673c57600。
本次为服务器最新实验代码和报告快照；在独立 publication 目录整理，原实验实现与 prism_pans 未修改。

## 本轮做了哪些

1. 新增 sharedkv_v22：以实际 prefix 读取误差构建 4×4 GQA 组代价矩阵，穷举 24 种一对一映射。K/V 使用同一映射。
2. 加入等条件同编号对照与搜索映射校准。64 fit 分为 48 edge-fit / 16 assignment-selection；独立 64 val 用于局部报告和 checkpoint 选择。两组均评估闭式初始化和 200 步蒸馏，主干冻结。
3. 在线只路由短 Q、prefix 输出及 LSE，不重排长 source K/V，不展开 4 KV heads 为 28；保留 private query/generated cache。
4. 新增带 head-map 的 bundle/fingerprint、兼容性检查、旧 v2.1 reader 桥接检查与真实 Qwen GQA CUDA 测试。
5. 新增质量、固定选择集合、最多 16 token decode、四轮配对 TTFT、resident 微基准、Nsight/VizTracer 及数据/物理共享审计。
6. 发布完整中文实验报告、轻量结果、八份校准报告和验证日志。完整逐请求 logits 与 profiler 原始文件继续保留在服务器。

## 结果与限制

- 18→19 映射为 [1,0,3,2]；26→27 为 [0,3,2,1]，数组含义为 target 组读取 source 组。
- 200 步后局部平均输出误差分别下降约 7.7% / 15.3%，但最坏请求误差没有同步改善。
- 175 条 TREC 回归上，18→19 从 68.00% 到 66.86%；26→27 从 72.00% 到 71.43%。
- 头对齐未显示稳定端到端加速。相同共享层对的 prefix KV 字节不因组匹配进一步减少。
- 175 条请求在 v2.1 已被观察，本轮不是新的未见最终测试。所有 reader 未通过原局部门限，仍为诊断候选。
- 报告保留不利结果、计时长尾和统计边界，没有将局部误差改善写成任务质量或部署认证。

## 文件与复现

入口为 scripts/run_sharedkv_v22.py；编译/校准为 scripts/compile_headalign_v22.py。
完整参数见 scripts/v22_*_jobs.json，方法说明见 README_HEADALIGN_V22.md。
根目录旧版本章节保留作历史参考；使用本分支时优先阅读当前说明。

沿用 v2 环境：Python 3.10、PyTorch 2.10.0+cu130、Transformers 4.57.6、vLLM 0.19.0；
vLLM 提供 FA2。Nsight/VizTracer 属于独立 profiler 工具。
模型、reader checkpoint、任务原始数据、prefix/payload cache、原始 profiler 和第三方工具二进制不纳入 Git。

服务器批处理脚本保留实测绝对目录；run_round4.py 仍使用 v2 固定模型路径和缓存目录，
不能仅设置 PANS_* 环境变量就迁移。移植需审计相应接口 hash，不能移除 guard。
在新目录先创建 logs/，准备原模型、数据、缓存和校准 reader；不要直接重跑已有输出的 joblist。
install_sharedkv_v22.py 用于向干净 v2 增加本模块；本分支已经包含该模块，无需重复安装。
finalize_v22_delivery.py 是原服务器交付脚本，不是安装步骤。

实测报告的 results/、audit/、assets/ 路径指服务器原目录；
精选快照分别位于 docs/layerkv_v22/results/、audit/、calibration/。
未发布聊天记录、登录凭据和原始逐请求数据；服务器私网地址在发布副本中省略。
算法源文件保持原样；发布 README 增加时效/路径说明，交付辅助脚本仅省略文档里的私网地址。
各文件源/发布 SHA256 记录在 audit/publication_source_sha256.json。

## 验证

原服务器完整测试：416 passed、5 skipped、87 subtests passed，保留一项既有 NumPy 只读数组警告。
本次发布副本另做 CPU 检查，结果记录在 validation/publication_cpu_tests.txt；没有重跑 GPU 实验。

本次发布 CPU 检查：397 passed、24 skipped、87 subtests passed；CUDA 隐藏，没有运行 GPU 实验。
v2.2 runtime guard 通过；32 个代码/测试文件与服务器逐字节一致。
另 1 个交付辅助脚本只省略其生成 README 中的私网地址，不改变实验逻辑。
