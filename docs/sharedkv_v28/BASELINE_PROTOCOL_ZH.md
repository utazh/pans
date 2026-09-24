# 跨层基线比较协议

## 方法名称

沿用已完成适配的 KVSharer-Prefix（作者yangyifei729/KVSharer），不要写成另一篇KVShare。xKV对应本地的xKV-CLF/P8前缀适配，而不是SR完整执行器。

KVSharer按每个prefix/预算的30条FIT查询搜索G24/G20/G16，不使用评估准确率挑图。原版校准提示语/短截断与这里固定历史prefix/full query不同，原模型支持也不同，必须写明。

xKV每四层拼接，固定缓存K逆RoPE后分解、V直接分解；P8选中行重建后加回原绝对位置。没有作者的SR landmark selector和SSD-native因子I/O。主线不更改selector，比较的是相同任务边界下的历史prefix表示。

## 数值检查与公平性

全秩FP64只用于验证线性代数、位置和运行接入；不能把全秩FP64当作高效部署模式。数值门限检查必须固定identity的真实选择集合，完整候选平均logprob最大差<0.05。

近似配置的质量本来允许变化；不要把近似低秩分解要求成全秩门限。FP32和BF16来自同一master、相同秩、U/SVh尺度。存储dtype改变也会改变重建运算精度，故表头应为“因子/重建精度”，而非声称只改变磁盘位宽。

排名要同时列秩、因子精度、source图、完整表示字节、理论选中行恢复量和实际部署路径。相同25% token预算或相同G，不能声称相同总字节。right因子/reader可常驻也不是免费，应另记内存和首次加载。

默认每格：I、R4、R4-trained→R0；KVSharer G24/G20/G16；CLF256/384和384/576，各FP32/BF16。未达G记录“未达到”，失败数值记录“接入未验证”，不用零或最佳替代值填表。

## 当前速度表的严格边界

本包只允许I/R4/R0进入正式cold/warm速度表。外部质量桥仍恢复完整原生payload，再使用常驻参考source/因子替换KV，没有合理的压缩I/O语义。

若论文需要外部延迟排名，后续最小必要增量是：

- KVSharer-Prefix：真实恢复其source在消费者选择集合上的并集，处理跨P8来源，private独立；不能强制改成我们的同IDs图。
- xKV-CLF/P8：持久化left按token块和right因子；真实读取left选中行，right常驻/冷加载边界一致，计入每层选中KV重建；然后与质量桥数值验证。
- identity也必须享有相同的文件组织、块粒度和通用I/O优化；否则不能把文件布局收益都算到压缩算法上。

上述SSD路径本包未实现。作者decode/throughput数字也不能直接复制到reprefill冷恢复表。可以先完成质量矩阵、主方法恢复矩阵，后续独立提交外部I/O适配，不为完成表格虚构速度。
