# 测试状态

日期：2026-09-18。

## 本地实际执行

环境：Linux，Python 3.13，PyTorch 2.10.0+cpu；没有 CUDA，没有 transformers，没有用户模型、prefix store 或 Pcache 数据。通过 GitHub 连接器静态读取了目标源代码；没有在此环境成功克隆并完整运行用户仓库。

执行：

```bash
PYTHONPATH=src pytest -q tests/test_sharedkv.py
```

结果：**24 passed, 2 skipped**。最后一次运行记录约 2.34 秒；两项跳过是 CUDA/FlashAttention 数值测试，不能计作通过。

已覆盖：流式 reference attention 对密集参考的因果/非因果比较；仿射读侧对显式重建 KV 的数学等价性；prefix key bias 对分支权重的影响；value bias 的质量加权；共同归一化；GQA 映射；RoPE split-half 交换结构；完整仿射拟合；query 度量 reduced-rank 约束与损失；output ridge；所有权图拒绝环/跨窗/leader consumer；mock loader 唯一物理恢复与 data_ptr 共享；不同选择集合拒绝；模拟 Qwen decoder/private cache fork；合成 BF16 store/trace 的完整编译及数据泄漏检查。

新 Python 文件另外执行了 py_compile 语法检查。

## 没有执行、不能宣称的结果

- 没有重新运行原仓库 307 项测试及相关子测试。
- 没有实际跑 Qwen2.5-7B、真实 DynamicCache/transformers 集成、真实异步 Pcache 或多任务缓存生命周期。
- 没有实测校准矩阵对真实模型的精度/困惑度/生成质量。
- 没有 CUDA/FlashAttention 数值实测，没有 GPU TTFT、实际 H2D 或 SSD 字节测量。
- 模拟 decoder 测试只验证接口设计和代数路径，不是实际 Qwen identity 验收。
- `accepted_local_only` 只表示编译器局部开发数据门限通过，不代表端到端非劣性。

真实执行前先按 README 的 original→identity→单层对 dense→多层 shared 顺序验收。全部拒绝或 TTFT 回退也是需要保留的实验结果。
