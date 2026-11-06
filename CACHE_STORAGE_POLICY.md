# 缓存存放约定
用户要求后续可再生成缓存和大型中间产物存放于服务器 /data1/pans/pans_kv。
本轮新目录与原路径映射见 /data1/pans/pans_kv/CACHE_LOCATIONS.json、audit/v28_storage_migration.json。
代码、模型、训练reader、原始结果和日志保留在既有位置。
/data1 为 HDD，不是 SSD。迁移前 SSD 计时不可和迁移后 HDD 计时混算；补测应建立新的存储条件与尝试记录。
