# v2.6静态审核与结果边界

审阅基准：11fde71c36ceecded60d560620900dec5c3a5d76。
通过GitHub连接器阅读当前脚本/控制/loader所有权代码；没有在本环境运行真实服务器或重新分析全部原始CUPTI记录。
报告全文作为用户给定证据。网络不支持本地git clone，因此CPU依赖测试使用此前提供的归档接口；安装器对服务器当前文件哈希逐项检查。

## 已确认

1. `scripts/v26_common.py::argv` 运行 mode identity/shared，同后端/P8/预算；`run_round4.py` 使用ProMixed、固定reuse与INT4 selector。I不是自动等同作者完整ContiguousKV。
2. `v26_training_worker.py::graph` 首先尝试旧图，不兼容时在P8内按相同IDs分组，并强制凑12个consumer。它不按评估质量选图，但不同预算确实改变source距离与fanout。
3. v25的RepairManager只加载原生小块，source按bank去重；不重算历史prefix。query和生成KV仍逐层独立。
4. v26 source_proxy在小区域使用原有同一套A/B/bias构建近似原生KV；它仍加载真实原生数据，所以不能把76.8%的质量结果直接变成省去该读取的速度结果。
5. 四任务每个单元都重训，不是把TREC权重搬到其他任务。然而历史TREC/p0/25%复用长期训练初始化，其余多数从prefix仿射拟合开始，训练历史是混杂变量。
6. 有些5%场景净子系统内存增加，不能只报告payload降低。

## 不可再保持的解释

“R4原生内容普遍不可替代”“普遍提高准确率”“全部预算都省总内存”“这些I/S结果已经超过完整ContiguousKV”“601项测试证明近似无损”。

## 新的代数诊断

在同Q、同私有KV、同所选位置、精确算术下，proxy区域与共享区域都来自同一仿射KV定义；共同softmax合并后等于固定reader的R0。
实际BF16前向的Q侧与K侧变换舍入点不同，后续selector也可能改变；只证明数学等价，不直接证明Qwen的bitwise或任务质量相同。

## 待验证原因，不是已知根因

- 固定G导致某些预算下较长层距/更多consumer，增加表示困难。
- fit≤240 token，但完整评估不限长度；长query泛化可能弱。
- 相同200步不等于与历史TREC相同初始化/训练总计算。
- teacher固定计划训练与在线student选择存在反馈差异。
- native和共享表示的接合可能影响分布，但proxy/R0结果未证明此因素占多大比重。

限定下一轮：固定checkpoint R0干预；相邻约束的G曲线；25/50%质量匹配操作点；外部表示基线。暂不改冷认证、晋升路径、P8评分公式、损失函数。
