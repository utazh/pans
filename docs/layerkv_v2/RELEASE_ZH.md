# LayerKV v2 发布说明

## 版本与范围

GitHub 分支及 VERSION 标记为 **v2**；内部模块版本为 **SharedKV v2.1**。
父版本：utazh/pans v1，commit 59312f2bd2504079fed03b53f283d8ade931fee2。
本次发布直接复制服务器已验证的实现；未修改原实验目录的实现文件或原 prism_pans 源码。
新增源码、脚本和测试与服务器交付 manifest 的 SHA256 一致。

## 内容

- src/contiguous_fuxian/sharedkv_v21/：读侧变换、联合归一化、元数据缓存、reader 预打包、hybrid attention、trace 和校准。
- scripts/*v21*：单次运行、顺序校准、质量评估、配对计时、Nsight/VizTracer 分析和审计；保留原实验脚本以便追溯。
- tests/*v21*.py：数值、缓存、数据隔离、统计边界与真实 GQA GPU 检查。
- upstream/pans_sharedkv_v21/：收到的原始代码包及其归属说明；实际服务器适配在主 src/scripts 中。
- README_LAYERKV_V21.md 与 EXPERIMENT_REPORT_V21_ZH.md：实测过程、结果、边界与复现入口。
- docs/layerkv_v2/results/、audit/、validation/：轻量结果、完整性证据与验证记录。

## 复现前提

先按根目录依赖和 docs/REPRODUCE_ZH.md 准备原系统。
v2.1 实测依赖：Python 3.10、PyTorch 2.10.0+cu130、Transformers 4.57.6、vLLM 0.19.0；
attention 使用 vLLM 提供的 FlashAttention2。无需单独安装 flash_attn。
VizTracer 与 Nsight 是可选分析依赖。

Git 仓库不分发模型权重、reader checkpoint、任务原始数据、prefix/payload 缓存、逐请求 logits、
原始 profiler trace 或 profiler 二进制。共享模式需要先按 v2.1 运行说明完成校准，
或使用服务器上已生成的 reader；当前 reader 不属于已认证部署版本。

原始 README 和批处理 shell 脚本保留实测服务器绝对路径。
在新机器上需修改模型、环境、数据路径和 GPU 编号；scripts/run_sharedkv_v21.py 是主入口。
批处理脚本属于原实验记录，不能在未准备依赖和路径时直接执行。
运行 suite 前创建 logs/；新输出目录不得与已有结果重名。
finalize_v21_delivery.py 是原服务器交付脚本，不是安装步骤。

报告中的 results/、audit/ 路径指原服务器实验目录。
本次发布的精选文件对应 docs/layerkv_v2/results/ 和 docs/layerkv_v2/audit/；
完整逐请求结果仍保留在服务器。
服务器历史验证为 380 passed、4 skipped、87 subtests；该数字不代表本次发布重新执行 GPU 实验。
本次发布副本的 CPU 测试结果单独记录在 validation/publication_cpu_tests.txt。

## 结果口径

质量来自独立 175 条 TREC holdout；正式 TTFT 来自同一 32 条回归 query 的四轮配对计时。
两类样本不可混算。共享节省的 3.56% / 10.68% 指唯一 selected prefix KV payload，
不代表整模型显存降幅。原系统到新执行路径的加速不能全部归因于跨层共享。

## 实测启动脚本与 v1 的差异

发布检查发现：原 v2.1 guard 将全部 8 项描述为 published-v1 接口，但其中
scripts/run_round4.py 的预期 hash 实际对应服务器实测快照；GitHub v1 的启动脚本已做路径参数化。
其余 7 个受保护库接口与 GitHub v1 一致。
本分支包含实测启动脚本原文，保持 guard 严格通过；未绕过检查或修改实验算法。
差异见 audit/server_launcher_vs_published_v1.patch。

因此，根 README 中历史版本的 PANS_* 环境变量示例属于 v1；本分支 run_round4.py
保留固定模型路径 /data1/llm/Qwen/Qwen2.5-7B-Instruct 和仓库内缓存默认目录。
直接复现实测需按运行说明准备该路径和数据。移植到其他路径需同时审计并更新接口固定 hash，
不能只改脚本然后跳过 guard。此版本的优先目标是发布准确的实验快照。

## 本次发布副本验证

- CPU：368 passed、16 skipped、87 subtests passed（CUDA 隐藏，未运行 GPU 实验）。
- 8 项实测接口 hash 全部通过；47 个 v2.1 交付代码文件 SHA256 全部匹配。
- 原服务器 GPU/全量测试历史日志一并保留，和本次 CPU 检查区分记录。
