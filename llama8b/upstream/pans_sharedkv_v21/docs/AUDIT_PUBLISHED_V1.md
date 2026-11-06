# 本次新发现及证据位置

代码来自分支v1/提交59312f2bd2504079fed03b53f283d8ade931fee2；main仍为e32c4a5。以GitHub connector读取并核对核心文件，未下载/运行用户服务器资产。

1. EXPERIMENT_REPORT_ZH.md：RTX3090、prefix4940、P8、25%预算；拟合/验证32/32、测试128。最佳单对26→27是末层；3对新加入3→2和18→19。因此不能把组合退化单独归因于误差积累。
2. docs/layerkv/results/comparison.csv：128在线identity/dense1为292.457/279.059ms；32固定为245.349/288.614ms。样本数不同、重复不足，方向相反。
3. comparison.csv：固定host payload 13,411,072→11,055,744 bytes，减少17.56%；唯一payload70,647,808→68,132,864 bytes，减少3.56%。都不是硬件H2D测量。selector5,690,880 bytes保持不变。
4. sharedkv/runtime.py RequestState.schedule_window只调度唯一source，decoder每impress_selection_period_size执行选择；P8并未减少成每组只载一份KV。原identity/共享均为两分支attention。
5. sharedkv/selection.py fixed_selection仅替换configure_impress_layer的IDs；run_sharedkv.py没有跳过configure_online_layer_selection。原固定集合仍支付选择器开销。上一轮旧v2的bypass replay不可直接拿来比较；本补丁已改为保留打分。
6. sharedkv/core.py vllm2每调用创建cuq、cuk两个CUDA arange；无共享也有56次attention，因而首prefill有112次长度数组构建调用。不是112次KV传输；也不是shared-identity差额的专属原因。新代码只加有界request-local长度数组缓存。
7. scripts/analyze_layerkv.py：KL是完整词表首token；准确率由scored_records中的标签续写评分。greedy一致率不是任务准确率。整份选择hash只有3/128相同不意味着token Jaccard只有2.34%。原分析全词表按存在的文件交集且silent skip，本补丁比较器默认拒绝不同集合。
8. sharedkv/runtime.py stats：CUDA峰值未逐请求重置；位于原greedy完成标签评分之后，不是TTFT内每请求峰值。父stats仍沿用命名，不将其当新证明。
9. compile/fit继承独立teacher-state拟合。原局部RRMSE0.1231，massP95=0.1403确实不满足0.03等门限。RoPE-only0.5605显著劣于dense；不应继续把纯RoPE块当首选表达形式。
10. 局部非劣门限是研究者指定目标，不是已验证的最终质量证书。新导出保持diagnostic_only，bootstrapping描述与小样本保守认证区分开。

本包的audit_data/published_selected_fields.csv是对公开comparison.csv选列的转录，不是新实验。recomputed.json只做算术重算。
