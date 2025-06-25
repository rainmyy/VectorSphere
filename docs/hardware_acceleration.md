# VectorSphere 硬件加速指南

## 概述

VectorSphere 支持多种硬件加速方式，包括 GPU、FPGA、持久内存 (PMem)、RDMA 和 CPU 高级指令集（如 AVX2、AVX512）等。通过合理配置硬件加速，可以显著提升向量搜索和计算的性能。

## 硬件加速器管理

VectorSphere 使用 `HardwareManager` 统一管理各种硬件加速器，主要功能包括：

1. 自动检测可用硬件资源
2. 根据配置初始化相应的加速器
3. 为不同的计算任务选择最佳加速器
4. 动态调整加速策略以适应负载变化

## 使用方法

### 1. 使用默认配置

```go
// 创建默认的向量数据库，会自动检测可用硬件
db := vector.NewVectorDB("vectors.json", 16)
```

### 2. 从配置文件创建

```go
// 从配置文件创建向量数据库
db, err := vector.NewVectorDBFromConfigFile("vectors.json", 16, "configs/hardware_config.json")
if err != nil {
    log.Fatalf("创建数据库失败: %v", err)
}
```

### 3. 手动创建硬件管理器

```go
// 创建硬件配置
hardwareConfig := &acceler.HardwareConfig{
    GPU: acceler.GPUConfig{
        Enable:  true,
        Devices: []int{0}, // 使用第一个GPU设备
    },
    CPU: acceler.CPUConfig{
        Enable:     true,
        NumThreads: 8,
        EnableAVX:  true,
    },
}

// 创建硬件管理器
hardwareManager, err := acceler.NewHardwareManager(hardwareConfig)
if err != nil {
    log.Fatalf("创建硬件管理器失败: %v", err)
}

// 使用硬件管理器创建向量数据库
db, err := vector.NewVectorDBWithHardwareManager("vectors.json", 16, hardwareManager)
```

## 配置文件格式

硬件配置文件支持 JSON 和 YAML 格式，下面是一个 JSON 配置示例：

```json
{
  "GPU": {
    "Enable": true,
    "Devices": [0],
    "CUDA": {
      "Enable": true,
      "MemoryLimit": 4096,
      "BatchSize": 128
    }
  },
  "CPU": {
    "Enable": true,
    "NumThreads": 16,
    "EnableAVX": true,
    "EnableAVX512": true
  }
}
```

## 支持的硬件加速类型

### GPU 加速

- CUDA：适用于 NVIDIA GPU
- OpenCL：跨平台 GPU 计算
- TensorRT：针对深度学习推理优化
- 多 GPU 支持：数据并行、模型并行等策略

### CPU 加速

- AVX/AVX2/AVX512 指令集
- 多线程并行计算
- NUMA 感知调度
- 缓存优化

### 其他加速器

- FPGA：可编程硬件加速
- PMem：持久内存加速
- RDMA：远程直接内存访问
- NPU：神经网络处理单元

## 性能调优

### 自动调优

VectorSphere 支持根据数据规模和查询模式自动选择最佳加速策略：

1. 小规模数据集（<10K 向量）：优先使用 CPU 加速
2. 中等规模（10K-1M 向量）：根据查询批量大小动态选择 CPU 或 GPU
3. 大规模（>1M 向量）：优先使用 GPU 或多 GPU 并行

### 手动调优

对于特定场景，可以手动调整以下参数：

- GPU 批处理大小：影响吞吐量和延迟
- 内存限制：控制显存使用量
- 线程数：根据 CPU 核心数调整
- 预取距离：影响缓存命中率

## 示例代码

完整示例请参考 `examples/hardware_acceleration_example.go`。

## 故障排除

### 常见问题

1. GPU 不可用
   - 检查驱动程序是否正确安装
   - 确认 CUDA 版本兼容性
   - 检查 GPU 是否被其他进程占用

2. 性能不如预期
   - 检查批处理大小是否合适
   - 确认是否启用了正确的指令集
   - 监控内存使用情况，避免频繁的内存分配

### 日志和监控

VectorSphere 提供详细的硬件使用日志和性能指标：

```go
// 获取硬件使用统计
stats := db.GetHardwareStats()
fmt.Printf("GPU 使用率: %.2f%%\n", stats.GPUUtilization)
fmt.Printf("内存使用: %.2f MB\n", stats.MemoryUsage)
```

## 最佳实践

1. 对于频繁查询的小型数据集，启用 CPU 加速和缓存优化
2. 对于大型数据集的批量查询，启用 GPU 加速并调整批处理大小
3. 对于混合工作负载，启用自动调优功能
4. 定期监控硬件使用情况，及时调整配置