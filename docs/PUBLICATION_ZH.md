# 发布版本说明

此发布副本在服务器上从第四轮最终实验目录整理。原实验目录、原始结果以及更早的 prism_max 目录保留。

发布版调整仅涉及：
- 新增 README、安装依赖、复现文档与结果入口；
- 最新运行脚本使用可配置的模型/数据/缓存路径，工作流使用当前 Python 和可配置的 Nsight；
- 3 项历史运行器测试改用临时夹具，去除对服务器真实数据目录的依赖；
- 收录基线参考源文件、分析脚本、轻量 CSV/JSON 汇总和 SVG 时间线。

核心 src 和 vendor Python 文件保持与实验版一致，哈希见 `reports/round4/published_core_sha256.json`。原实验完整源文件快照见 `original_source_sha256.json`，两者范围不同。

历史 Markdown 报告保留当时的服务器路径和文件定位，因此其中指向 audit、results、profiles 的部分路径不是公开仓库路径。当前可访问的精简结果统一在 `reports/round4/`；原始 Nsight/VizTracer 文件、逐条样本记录、模型、数据、缓存、旧日志与虚拟环境留在实验服务器。

`audit/round4_baseline` 是第三轮 src/vendor 参考快照，用于同期对照；`audit/round3_baseline` 只包含回归测试需要的旧实现。它们不是原始模型或数据。

依赖清单为参考环境约束，不是整个操作系统与 GPU 驱动的完整锁文件。新机器上的 TTFT 不能直接与共享服务器的历史测量等同。
