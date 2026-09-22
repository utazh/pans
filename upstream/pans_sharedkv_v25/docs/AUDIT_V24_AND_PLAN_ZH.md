# v2.4审核与v2.5研究边界

## 已核对的材料

- GitHub v2.4：a6be130c921024a4cd358ccf5841e4a46f547ae2。
- 用户提供的 EXPERIMENT_REPORT_V24_ZH.md（报告本身不是本地复现实验）。
- 当前 sharedkv_v24/train.py、replay.py、faithful.py、records.py、topology.py；v21 runtime/ops；Pcache提交、resolve与来源统计接口。
- DeepSeek-V4.1报告第2.3.1、3.1.2、3.2节及官方config；报告Figure 4以页面图核对。

本包仅新增模块。测试不是完整Git仓库/GPU实验：在最小接口夹具中使用已核对的辅助模块，原Pcache及完整runner采用接口模拟；服务器需执行原有完整回归。

## 当前证据

三种子开发集非劣条件均未过，必须保留该结论。报告同时说明，未通过不是已证明真实退化超过1pp。不能把该认证条件变成禁止研究任何更低G的永久条件。下一轮另登记有限探索；探索与最终认证分开，不回写v24结果，不挑已观察final768的最好seed。

报告的两种SSD冷策略均有延迟下降证据，但每策略4轮×同32请求，不是完整500条冷计时。warm两个进程设计方向相反，仍不具备稳定warm加速结论。prefix所选7.12%、加入reader/packed后5.99%是局部资源账，不是模型总内存或SSD压缩率。

v24服务器修复尤为重要：用实际FA2前向、packed布局和逐token续写消除了旧参考回放差异。本包直接复用该faithful AttentionVJP；不能退回上传原型中不等价的全序列参考前向。反向仍是数学softmax的FP32 VJP及BF16 STE，不是对有限精度硬件运算精确求导。

## DeepSeek启发与不等同之处

DeepSeek共享的是global KV和可选的index，所有层仍有本层Q与本层SWA KV；共享架构参与训练。其40层模型主KV来源在2/8/14/20，index来源更多。官方配置滑窗128。

本包只借鉴“共享大记忆、保留小的层专属记忆”的分工，不实现SWA，不复制CED，不使用Bounded Replay。当前dense Qwen旧prefix KV已经离线计算完成；原生补丁来自持久化的本层KV，不重新prefill旧prefix。

## 改动

原选择集I按token位置有序。取最后r个已选完整物理块为R，其他为C：I=C不交并R。r=0是原纯共享控制；默认新增探测r=4。R不是绝对位置最后64个token，也不保证这些token连续。不能把原来未选择的token加入R。

consumer读取source的C视图，Q经A、prefix聚合经B；R用consumer原生KV并与private tail合成一个native分支；两分支共同归一化。无double count，不重排/重建长source KV。source本人仍读取完整I；因此资源账必须数full owners和consumer patches两部分。

离线来源图和r策略固定。在线P8继续依赖当前query选择ID；按既定r划分后只拉取所需source及patch。A/B同v24共同优化，未新增loss。

## 协议

1. 在fit的实际ID合约下登记G26/G24/G22/G20/G16/G14，先跑G24/G22的r=0/4。
2. 相同seed24与初始化、同训练样本/目标/步数；记录实际训练工作量。
3. 预登记探索分区与最终标准。旧final768是已观察材料，不能再作为新最终集。
4. 收集开发结果后可以拒绝不合格候选，不能自动认证；不要无限调r直到旧数据过关。
5. 热缓存、SSD-cold/native、SSD-cold/preserve都保留同执行控制；每arm独立进程，继承v23认证。
6. 真实硬件审计native补丁额外请求数/晋升/数据流，不能只看source数量。

## 风险

少量原生块未必能修复主要误差，当前没有Qwen结果支持。r=4增加磁盘chunk请求、拼接和元数据处理，可能使TTFT变差。consumer修复载入后只更新修复块评分，否则整层晋升会吞掉节省。v25没有优化重复晋升或重写cache策略；此类通用优化需要identity/shared共同对照。

方法仍绑定模型、prefix、位置、token预算和校准数据。无标签训练不等于无训练，也不等于任意新prefix无成本。完整参数指纹、fit/val角色和评估ban必须保留。

## 实施范围

支持batch=1、Qwen2结构、full attention、固定RoPE、相同P8窗口/相同选择ID、FP16 payload/BF16 compute、单GPU。未提供tensor parallel、device_map、多请求batch、动态RoPE、持久文件删除/压缩或生产部署认证。
