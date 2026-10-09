# v6 服务器快照使用说明

根目录代码对应服务器 src 的 Qwen7B 分支。overlays/llama31 是 src_l8 相对共同基线的完整 12 文件差异，包含未提交的 --conds 改动；guard 与运行代码配套保存。

复现 Llama 时，在**独立的 v6 checkout** 中执行：

    cp -a server_a100/overlays/llama31/. ./

此操作覆盖该独立 checkout 的对应文件，不应对正在运行的 Qwen 工作目录执行。Qwen 的新 geometry 警告不合入 Llama 原实验代码。Qwen14B/32B 精确历史代码使用 v5，本轮没有更新其核心源码。

env_*.sh、xfer/ 与 analysis/ 保留服务器绝对路径与脚本。迁移前更新根目录、模型、Python、数据及缓存路径，并检查 GPU/CPU 绑定。先计时环境后模型环境，例如先 env_timing.sh 再 env_l8.sh。历史队列与停止/清理脚本只用于来源记录，不建议整目录自动执行。

主要入口：

- scripts/v29m_pipeline.py：collect/train/derive/dev/test，参数以脚本为准。
- xfer/convert_llama.py：创建纯续写格式数据副本。
- xfer/mlr_build.py：Wiki/其他任务闭式 reader。
- xfer/train_multi_v24.py：留一任务多前缀 reader 训练。
- xfer/conc.py / conc_report.py：多进程、多 GPU 冷恢复压力实验。
- scripts/v29m_timing_sys.py：Llama overlay 增加 --conds preserve 或 --conds warm。
- xfer/timing_report.py：默认剔除未解决标记块；--all 仅敏感性分析。

结果、数据使用边界与两项 Llama 配置测试失败见 [v6 报告](../docs/layerkv_v6/RELEASE_ZH.md)。模型/数据/adapter checkpoint/KV 不在仓库内。上传过程没有修改原实验代码。
