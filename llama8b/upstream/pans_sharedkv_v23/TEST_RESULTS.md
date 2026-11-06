# 本次实际测试范围

运行命令：

```text
PYTHONPATH=<本包src>:<此前附加包的纯CPU依赖src> python -m pytest -q tests/test_sharedkv_v23.py -rs
```

**37 passed，2 skipped。**完整输出见 `validation/cpu_pytest.txt`，环境见 `validation/environment.json`。

## 通过的内容

- 一source多consumer结构、P8 leader与原生例外、14/7/4来源规划、非法跨窗/所有权拒绝。
- FP64下：读侧仿射运算与显式构建近似KV一致，包含K偏置、V偏置、GQA、因果private tail和共同归一化。
- 真实PyTorch toy多层模块：所有reader得到梯度，冻结主干无参数梯度；多个consumer传入完全相同的source tensor指针。
- 合成组优化能降低定义的组损失。
- 完整query和候选标签前缀捕获/重组，与逐步执行在toy模型上数值一致；自由生成与不合规capture拒绝。
- 合成raw BF16 prefix store、toy trace、mock模型加载/身份验证、两步组训练、最终head KL、候选checkpoint导出的闭环。
- 带有真实职责的模拟CKLFU驱逐/owner计数重置、旧异步工作/未持久化数据拒绝、cold首次decoder I/O范围与label scoring隔离。
- 实际Linux mincore系统调用、文件所有权/链接/路径限制、页未驱逐时fail-closed。
- 读时uncache策略的线程可见性、非目标文件不受影响、异常退出恢复、page-hot/native不合法组合拒绝。

## 两项跳过的原因（不是两个都因为没有GPU）

1. 本环境文件系统在对测试文件执行DONTNEED后，8/8页仍驻留。真实页驱逐认证因此跳过，程序的正常行为是拒绝将其记为SSD冷测量。本次没有成功认证SSD冷I/O。
2. PyTorch为CPU构建，没有CUDA设备，因此真实CUDA数值测试跳过。

## 尚未测试或不能宣称的内容

- 没有下载/运行真实Qwen2.5-7B权重、用户数据、Pcache、reader或完整416项服务器回归。
- 没有RTX3090/真实vLLM后端多消费者端到端质量、峰值内存、TTFT或Nsight新结果。
- 没有认证用户SSD控制器状态、真实物理磁盘I/O量或任意文件系统都支持定向page eviction。
- 本包不能据此宣称新的质量保持、压缩率部署可用性或加速结果。

代码是对最新已发布接口的独立附加实现。服务器需要先核验依赖、接口guard、GPU identity replay、many-to-one alias与cold前置条件，再运行完整组训练和配对评测。所有候选默认不自动部署认证。
