# 实际测试状态

本轮本环境执行：**40 passed, 2 skipped in 2.90s**。日志：`validation/pytest_cpu.txt`。
环境详情：`validation/environment.json`；PyTorch 2.10.0+cpu，无CUDA。

## 已执行

- 评分记录schema、query hash、fit/validation隔离、平均token logprob与sum的区别。
- 28层结构候选、77/78式不一致集合、P8 leader保护、一跳所有权与逐阶段source数。
- 全网络私有候选prefix重算；包含标签末token；query与候选预测位置没有off-by-one。
- 两个reader联合梯度、冻结backbone参数无梯度。
- 两遍逐分支VJP反传与一次性完整图的梯度对照，含activation checkpoint开/关两组。
- 数据记录→source-plan→初始化→两步合成训练→BF16 reader导出的合成闭环。
- 候选margin的代数充分条件。
- 实际greedy关键字参数形式的模拟采集；无gold标签读取。
- 预取/晋升事件归属的模拟检查，正常和异常退出恢复原方法。
- 保守非劣计算，不使用零翻转导致退化的bootstrap作为零风险证明。

## 跳过/未执行

- 两项CUDA测试因环境无GPU跳过。
- 未运行真实Qwen2.5-7B、vLLM FA2、RTX3090训练显存验收或新方案准确率/TTFT。
- 未重新运行用户服务器461项完整测试。
- 未执行SSD mincore认证，也未修改原v23冷缓存算法；新的读取审计仅做模拟测试。
- 实际source/reader数、目标数据集是否能生成g14/g16，以及开发集是否通过，都等待服务器实测。

## 源码核对与测试底座

GitHub连接器已实际读取v2.3 `3868c6b64332f89cc0b301c75b53c35630302f0e` 的训练、capture、block、runner和恢复接口。直接网络clone受限，所以数学测试使用先前交付包里的依赖模块；`sharedkv_v23/block.py`、`sharedkv/io.py`、`sharedkv/fit.py` 与本次读取的对应Git blob hash一致。
这不等于在完整v2.3 checkout上完成了集成验收。安装器会在用户独立worktree上检查9个真实接口；不一致就拒绝，不绕过guard。

本轮首次pytest因测试底座Python包路径配置未包含新模块而收集失败，调整仅测试用PYTHONPATH/复制新模块后重新执行；没有删除失败测试。最终日志是完整新测试集。
