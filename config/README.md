# VectorSphere 自适应配置说明

## 概述

VectorSphere 支持自适应配置功能，可以根据系统硬件状况、数据规模和性能要求自动调整索引策略和硬件加速设置。

## 配置文件

### adaptive_config.yaml

主要的自适应配置文件，包含以下配置项：

#### 基础配置
- `config_file_path`: 配置文件路径
- `enable_hot_reload`: 是否启用热重载
- `hot_reload_interval`: 热重载检查间隔（秒）

#### 自适应开关
- `enable_adaptive`: 总开关，控制是否启用自适应功能
- `enable_hardware_adaptive`: 硬件自适应开关
- `enable_index_adaptive`: 索引自适应开关
- `enable_strategy_adaptive`: 策略自适应开关

#### 硬件自适应参数
- `detection_interval`: 硬件检测间隔
- `gpu_memory_threshold`: GPU内存使用阈值
- `cpu_usage_threshold`: CPU使用率阈值
- `enable_auto_fallback`: 自动回退开关

#### 索引自适应参数
- `window_size`: 性能记录窗口大小
- `optimization_interval`: 索引优化间隔
- `min_samples`: 进行优化所需的最小样本数

#### 策略自适应参数
- `performance_weight`: 性能权重
- `quality_weight`: 质量权重
- `context_similarity_threshold`: 上下文相似度阈值

## 使用方法

### 1. 启用自适应功能

在创建 VectorDB 实例时，系统会自动尝试加载配置文件：

```go
db := NewVectorDB("data.db", 10)
// 系统会自动加载 config/adaptive_config.yaml
```

### 2. 手动加载配置

```go
err := db.LoadConfigFromFile("config/adaptive_config.yaml")
if err != nil {
    log.Printf("加载配置失败: %v", err)
}

// 应用配置
db.ApplyAdaptiveConfig()
```

### 3. 启用热重载

如果在配置文件中设置了 `enable_hot_reload: true`，系统会自动监控配置文件变化并重新加载：

```go
// 启动配置热重载（在 NewVectorDB 中自动调用）
go db.StartConfigReloader()
```

### 4. 保存配置

```go
err := db.SaveConfigToFile("config/adaptive_config.yaml")
if err != nil {
    log.Printf("保存配置失败: %v", err)
}
```

## 自适应策略

### 硬件自适应

系统会根据以下因素自动调整硬件使用策略：
- GPU 可用性和内存使用情况
- CPU 使用率
- 系统负载

### 索引自适应

系统会根据以下因素自动选择最优索引策略：
- 数据规模（向量数量）
- 向量维度
- 查询模式
- 历史性能表现

### 策略自适应

系统会根据以下因素动态调整搜索策略：
- 查询质量要求
- 性能要求
- 上下文相似度
- 历史查询模式

## 监控和调试

### 获取性能洞察

```go
insights := db.GetPerformanceInsights()
fmt.Printf("性能洞察: %+v\n", insights)
```

### 查看自适应状态

```go
status := db.GetGPUAccelerationStatus()
fmt.Printf("GPU加速状态: %s\n", status)
```

## 注意事项

1. **配置文件格式**: 必须使用有效的 YAML 格式
2. **权限要求**: 确保应用有读写配置文件的权限
3. **资源监控**: 启用自适应功能会增加一定的系统开销
4. **GPU 要求**: GPU 自适应功能需要系统支持 CUDA
5. **热重载**: 频繁的配置变更可能影响系统性能

## 故障排除

### 常见问题

1. **配置文件加载失败**
   - 检查文件路径是否正确
   - 检查 YAML 格式是否有效
   - 检查文件权限

2. **GPU 加速不可用**
   - 检查 GPU 驱动是否正确安装
   - 检查 CUDA 环境是否配置正确
   - 查看系统日志获取详细错误信息

3. **自适应功能不生效**
   - 确认 `enable_adaptive` 开关已启用
   - 检查相关子开关是否启用
   - 查看日志确认配置是否正确应用

### 日志查看

系统会在日志中记录自适应配置的加载和应用过程：

```
[INFO] 成功加载配置文件: config/adaptive_config.yaml
[INFO] 自适应配置已应用: 硬件自适应=true, 索引自适应=true, 策略自适应=true
[INFO] 配置热重载已启动，检查间隔: 30秒
```