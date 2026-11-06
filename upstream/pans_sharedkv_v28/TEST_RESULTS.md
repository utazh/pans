# 本环境验证范围

CPU pytest：**51 passed，1 skipped**。CUDA不可用，因此GPU精确SVD测试跳过。

已执行：
- 冻结R4→R0的A/B/bias和身份保护；拒绝参数、图、禁用名单变化。
- 同一FP64 master导出FP32/BF16、U/SVh尺度、零奇异值、全秩/截断重建和RoPE回转。
- 参数化v27 KVSharer距离与来源remap的玩具数值检查。
- 六序列位置平衡、不同计时块、嵌套CLI、外部质量桥禁止冷计时。
- 固定评分门限、按块统计、source并集与right因子独立资源记账。
- 16格合成文件端到端prepare、拟合/验证隔离、frozen checkpoint导出与SHA。
- 模拟子进程成功/冷认证失败/不可覆盖/成功续跑与篡改拒绝。

测试使用挂载的既有代码包依赖；模型、tokenizer和进程处使用合成/模拟对象。没有执行真实Qwen、vLLM CUDA、SSD/mincore、KVSharer完整模型搜索、16格实际评估或硬件计时。没有在这里重跑服务器656项回归。

新增CLI帮助及全部Python文件语法也经过检查。安装器设计为对实际v2.7七个接口hash校验，不跳过服务器核验。

数值gate .02/.05沿用v27，不修改；新报告没有质量或速度认证。
