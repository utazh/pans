# SSD冷恢复协议与安全边界

## 被测对象

本协议测“已预计算的prefix，KV主体在持久化Pcache文件中，但在请求开始前不驻留于应用GPU/CPU缓存，也不驻留于OS文件页缓存”。模型、reader、K4索引、文件元数据和算子保持热。不能将此称为完整服务冷启动或首次生成prefix。

所有方法使用当前CKLFU、同P8 selector、同预取方式、同物理chunk尺寸。程序不改变SSD文件格式、不新增O_DIRECT路径、不删除文件。

**读取时DONTNEED必须控制：** 原vendor.uncache在 `_chunk_source_readonly` 等磁盘分支每次读取前调用DONTNEED。默认native策略保留它；page-hot不允许native，否则page刚读热就又被驱逐。控制OS page影响时，identity/shared的page-hot/ssd-cold四组都显式使用 `--read-advice preserve`。该策略仅在请求期间对核验过的目标payload文件停止重复uncache，其他路径不变，线程完成后恢复函数；记录抑制调用数。它是共同I/O实验策略，不得记成共享的算法贡献。

## 为什么不是 --warmup 0

1. add_task/insert初始化本身可能填入驻留数据、触碰文件页。
2. 32个同prefix请求中，第一条之后即可变热。
3. 取消模型/库预热会混入lazy初始化、内存分配首次成本。
4. torch.cuda.empty_cache不清除还被cache对象引用的KV。

## 清理过程

1. 确保新loader还没有payload pending/loaded状态，等待旧prefetch worker队列与CUDA活动完成。
2. 检查真实cache token均有已存在文件，避免to_disk写新文件。
3. 使用ScoreUnit.evict_gpu/evict_cpu正常维护堆与容量；不直接清tensor而让活跃堆损坏。清理所有应用payload及累计cache score，K4索引的独立CPU副本仍常驻。
4. 真实目标路径来自当前loader的key_tokens/value_tokens.token.path。不是原始BF16 prefix store，也不是猜测文件名。
5. 对目标prefix的所有层payload文件使用POSIX_FADV_DONTNEED，之后逐文件mincore核验。
6. 所有resident pages为0才进入计时，失败写invalid_cold_precondition并中止。整个过程在原TTFT外，并单独记录setup_ms。
7. 下一个正式query重新做一次。不能用第一次的冷状态代表整组。

## 系统/文件限制

- Linux、本机块设备支持的文件系统；确认为目标SSD/NVMe，而不是tmpfs或网络文件系统。可使用findmnt -T和lsblk核对挂载来源和设备类型，保存结果，不据文件扩展名推断介质。
- cache文件必须是用户拥有、非symlink、非hardlink的独立基准副本；禁止与其他任务共享同一批会被驱逐的inode。
- fadvise只是建议；脏页、活动映射、文件系统策略可能使它无法驱逐。正常准备并同步独立缓存副本后再实验；工具不自动改系统全局策略，不循环洗出假“成功”。
- mincore只是当时快照，不是下一时刻页面状态保证。独立副本、无并发同文件reader是前提；文件身份/size/mtime在准备前后检查。
- 不清drop_caches、不强制全局sync、不操作SSD控制器缓存；底层存储DRAM/闪存缓存状态不在认证范围。

## 三种对照

主对照包括warm/native、ssd-cold/native，以及page-hot/preserve与ssd-cold/preserve。最后两组在相同read-advice下隔离OS页冷热。warm/native与page-hot/preserve还改变了读取建议策略，不应视为单因素对照。每一种都必须同时运行identity和相同读取路径的shared；P8 selector和预取都保留。

即使SSD冷没有加速，也是真实结果。小文件数量、页读放大、cache promotion和prefetch重叠可能决定效果；不能只用节省的字节除带宽当作TTFT预测。

## 记录与解读

- application_payload_empty、gpu/cpu_mib_after、target payload chunk数。
- page_before/page_after、全文件总字节、resident pages；SSD冷需要page_after.resident=0。
- 原runner logits-ready TTFT，完整评分结果在scored_records。
- 首次decoder的/proc/self/io read_bytes/rchar和major/minor faults变化；不会把label scoring算入。
- 进程read_bytes反映存储层读取记账，不能等同于PCIe或SSD硬件NAND；并发storage流量另外控制。
- 硬件H2D/异步重叠用独立Nsight，正式计时不启用profiler。
- 当前实现reader/index常驻，冷加载它们的完整服务实验不在本包实现中。报告其常驻字节、离线构建成本；不能把主体冷恢复结果包装成完整冷启动。

## 本容器测试的已知限制

测试环境实际mincore系统调用可用，但对测试文件调用DONTNEED之后页面仍驻留，真实页驱逐认证测试因此skip。fail-closed逻辑的模拟测试通过。在服务器上必须先用独立小文件验证该能力，再测真实Pcache；若不支持，不能声称本工具已经获得SSD冷结果。
