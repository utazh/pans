# 本次实际验证范围

- `33 passed, 2 skipped`，完整输出见 `validation/cpu_tests.txt`。
- 两项跳过分别需要 CUDA + 独立 FlashAttention、CUDA + vLLM FA2。当前环境没有 GPU。
- Python 3.13.5，PyTorch 2.10.0+cpu。
- `compileall` 通过；run/refine/audit/bench/compare/slice 六个 CLI 的 `--help` 通过。

测试覆盖读取矩阵恒等式、联合归一化、GQA、RoPE、梯度、合成拟合、reader/bundle 继承与门限、mock decoder/private cache/物理别名、metadata cache、固定IDs仍执行selector、窗口与按需调度、shadow加载，以及保守有害翻转统计。

测试依赖本地还原并核验 Git blob SHA 的已发布 `sharedkv` 数学/读取接口。不是对整仓库342项服务器测试的复跑；后者属于用户报告，不计入本包测试数。没有模型权重、原始运行日志、真实calibration traces或RTX3090访问。

尚需服务器验证：vLLM ABI与LSE数值一致性、真实Qwen prefill/decode/label fork、host来源计数、真实H2D、端到端质量与重复TTFT。metadata/packed/hybrid均为待测优化，未作性能承诺。
