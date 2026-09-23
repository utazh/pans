# 实际测试范围

本环境实际执行：

```text
52 passed, 2 skipped in 5.11s
```

环境：PyTorch 2.10.0+cpu，SciPy 1.17.0；CUDA不可用。

通过范围：
- 原生repair对照与原数学路径bitwise相同（CPU FP32）。
- source-proxy在CPU FP64中与显式仿射共享完整attention一致，包括GQA、K/V偏置、joint softmax、prefill与decode几何。
- 原生patch预算、原始source不可变、禁止完整prefix重建。
- 完整/短末块注册，FP16类型与持久化约束，重复和部分chunk拒绝。
- 准入clone拥有独立storage，修改request source不会改变缓存clone。
- 非GPU准入不在生产路径静默启用；模拟对照覆盖回退与API恢复。
- 上层wrapper默认结果原对象传递、异常时恢复patched函数和请求锁。
- 固定结构直接共享控制，不修改原checkpoint，保留政策和泄漏禁用名单。
- 原v25非劣公式数值复现：R4 seed25下界约-2.982762pp，仍未通过。
- 不同query block、平衡位置、四层launcher参数、重复请求不增加独立质量样本。
- JSON输入、重复记录、错误配对、非有限计时拒绝。

跳过：
1. CUDA准入clone的真实device行为。
2. vLLM FA2下source-proxy诊断执行。

没有验证：
- 真实Qwen/RTX3090，任何新准确率、SSD/H2D字节或TTFT。
- 真实Pcache ScoreUnit完整堆不变量和GPU内存生命周期。
- 全部547项既有服务器测试。
- 同一服务器上的多进程并发（本包逐臂新进程、单请求串行）。
- 14B、多卡、新prefix泛化、完整外部baseline。

读取仓库通过GitHub连接器完成；容器不能直接克隆GitHub。CPU依赖从本会话此前代码包恢复，实际消耗的v25/policy.py、attention.py、runtime.py、replay.py、train.py和当前sharedkv/core.py均核对Git blob内容身份。测试以数学与模拟接口为主，不冒充完整v2.5 checkout的端到端验收。

所有source-proxy运行只能作质量诊断；reuse-prefetch须在用户服务器通过完整GPU数值/缓存/冷态/文件审计后才允许进入正式速度研究。没有新增部署认证。
