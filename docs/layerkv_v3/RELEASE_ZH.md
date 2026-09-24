# v3 发布与复现说明

v3是服务器v2.8实现和当前已完成实验的公开快照，基于v2.7；内部sharedkv_v28名称保持不变。实验源码/测试/协议按服务器版本发布，原实验目录和prism_pans不修改。当前不是全部实验完成版：48个主质量臂和112个外部质量臂完成，计时39臂成功、1臂失败。

入口为 scripts/matrix_v28.py；子命令和参数可用 --help 查看。配置模板为 configs/v28_spec.json，实际使用前须检查路径与设备。prepare生成本地study manifest；每台设备的独立质量配置覆盖需由准备后的manifest复制生成，原脚本中的configs/v28_external_manifest.json是服务器输入，不随轻量发布。所有作业和结果应使用新的独立输出目录。

模型、数据集、训练reader、prefix/K4索引、KV缓存、完整因子/来源张量、原始逐题记录及大型运行manifest留在服务器。原报告中的results/v28_summary_current内容在本目录results/；完整阶段的results/v28_summary尚未生成。本文不保证空克隆无需输入即可执行模型评估。

docs/sharedkv_v28是随输入代码包提供的方案材料，未执行的计划不等于完成结果；以根目录EXPERIMENT_REPORT_V3_ZH.md为准。服务器完整回归与本次CPU发布测试分别保存于validation/。

已完成冷测的介质是Intel SSD。复现前应确认缓存文件实际所在介质，HDD数据不能当作SSD冷恢复。严格冷认证失败记录保留，不放宽门限、不补造缺失计时。

已知服务器 scripts/v28_final_results_audit.py 的private长度q+1断言不适用于全部任务。当前结果采用 query长度+最长标签续写长度−1 的只读复核，见audit/v28_quality_and_partial_timing_audit.json。该脚本未执行，不是本轮计时停止原因；保留历史实现与此限制，发布不擅自改动实验逻辑。v28_write_report.py预期完整研究输出，不能用于覆盖本次部分结果报告。

输入数据和模型不随Git分发；本文不包含新增论文。外部基线的适配与运行边界仍适用，不能把质量桥的TTFT列为KVSharer/xKV官方系统速度。

发布副本CPU回归：680 passed、33 skipped、87 subtests passed；58个源文件哈希与服务器对应文件一致，19个v28 Python文件语法检查通过。此验证未启用CUDA，也未重跑模型实验。
