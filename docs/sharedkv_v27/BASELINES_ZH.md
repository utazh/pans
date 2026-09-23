# 外部基线与适配边界

## 为什么单位reader不是KVSharer

固定我们方法选择的图后设置单位A/B，是“读侧补偿”的消融。KVSharer还要进行跨层距离排序和最终隐藏状态相似性搜索；不能用16.8%代表作者方法。

作者仓库： https://github.com/yangyifei729/KVSharer
核对主提交：39e5c19176940f8d0089829fa8aab7992f85cb37。
入口：test_llama.ipynb、test_internlm.ipynb；实际缓存类在llama_real_share/cache_utils.py，模型适配在modeling_llama_kvsharer.py。

v27的搜索函数是一份独立的算法逻辑参考，不是作者整个模型适配。外部正式基线分两步：
1. 作者原协议的原支持模型冒烟，验证搜索和结果读取没有误解；不把其decode速度用于我们的reprefill表。
2. 显式命名KVSharer-Prefix适配：只共享历史prefix，private query/生成KV原生；原距排序/隐藏状态门限保留；搜索数据与评估隔离；层/位置兼容限制全部公开。

不允许：以我们训练得出的图代替作者搜索；把不同token集合取交集却不披露；直接复用Llama图到Qwen；伪造达到目标来源数（未达到就报告实际值）。

## xKV / xKV-SR

作者仓库： https://github.com/abdelfattah-lab/xKV
核对主提交：05d91ecb0d698279aa4220fda5d7a5108036d692。
已发布SR与效率代码，不能再说“只有全量重建实现”。
入口：evaluate/eval_acc.py；examples/xKV；examples/xKV_SR；efficiency。
Qwen作者README要求--flash2。

示例仅用于作者场景的复现，不是本项目冷恢复速度：

```bash
python evaluate/eval_acc.py \
  --model_name_or_path /path/to/model \
  --dataset_name ruler/vt --datalen 65536 --flash2 \
  --method xkv --merge_k --merge_v --rank_k 384 --rank_v 576 \
  --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1
```

v27提供的是xKV-CLF+P8质量参考：按论文共享token basis，K在逆RoPE空间分解，重建选中行。不包含作者landmark选择或优化SR后端，也不在SSD存储路径上读取因子。
正式性能适配仍需：持久因子schema、cold-ready因子文件、所选行并集读取、CPU/GPU驻留、重建workspace记账、与作者核的数值对照。完成之前，参考质量行不得附上本项目原loader的TTFT。

## ContiguousKV

论文：arXiv:2601.13631。应把完整方法/忠实复现单列，说明索引复用、跨周期预取和缓存评分设置。当前同后端P8 identity是隔离本方法增量的强控制，但名称不能自动换成ContiguousKV。
对外部方法及我们：相同prefix、非共享query、权重、精度、字节计量、冷热条件和TTFT边界。

## 比较表必须有两种预算

1. 相同token支持预算：隔离跨层表示误差与恢复量。
2. 相同净资源预算或相同质量容差：防止“少读很多但误差更多”被写成普遍更强。
计入A/B、packed、xKV左右因子、native小块、临时重建、索引和工作区；不把请求缓存减少写成SSD文件压缩。
