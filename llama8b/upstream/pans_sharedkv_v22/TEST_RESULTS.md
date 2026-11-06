# 实际测试范围

29 passed, 2 skipped（CUDA 不可用），日志见 validation/cpu_tests.txt。

通过的内容：读侧头映射与显式仿射 KV 参考计算的一致性；K/V 原始对象不变；GQA/private tail/causal/RoPE；非法映射拒绝；赋值问题及身份映射 tie-break；query 级切分；包含映射的 fingerprint；合成植入排列恢复；合成 prefix-store→16边拟合→1步梯度→bundle 输出闭环；预算计算；临时兼容视图不改输入文件。

这些是单元测试与合成数据测试。运行时桥接测试显式使用父接口模拟，不是完整 Qwen 执行。测试依赖来自会话中已有的旧版本包；本轮 GitHub 已读到的 refine.py 与 ops.py blob 身份和本地复用代码一致。远端仓库代码通过 GitHub connector 检查，容器直接 git clone 因 DNS 不可用没有成功。

没有运行：RTX 3090/vLLM FA2 真机数值核验、真实 Qwen accuracy、TTFT、实际 H2D、380项原工程回归测试。CUDA 两项测试已提供给服务器运行。不得把本地29项与服务器380项相加宣称已完成新版本全套验证。

新的 read-side 头映射可能降低质量也可能不改善，可能增加在线 gather 延迟。需要先做身份映射控制、独立验证及配对性能测量。
