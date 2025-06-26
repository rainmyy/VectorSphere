# VectorSphere 硬件加速搜索指南

## 概述

VectorSphere 支持多种硬件加速器，包括 GPU、FPGA、RDMA、PMem 等，以提高向量搜索的性能。本指南将详细介绍如何配置和使用这些硬件加速器，以及如何根据不同的工作负载类型选择最佳的加速策略。

## 支持的硬件加速器

VectorSphere 目前支持以下硬件加速器：

| 加速器类型 | 常量值               | 适用场景          | 优势           |
|-------|-------------------|---------------|--------------|
| CPU   | `AcceleratorCPU`  | 通用场景，小规模数据    | 无需额外硬件，兼容性最佳 |
| GPU   | `AcceleratorGPU`  | 大规模并行计算，高维向量  | 高吞吐量，适合批量处理  |
| FPGA  | `AcceleratorFPGA` | 低延迟要求，特定算法优化  | 超低延迟，能耗效率高   |
| PMem  | `AcceleratorPMem` | 大规模数据集，内存受限环境 | 大容量，持久化存储    |
| RDMA  | `AcceleratorRDMA` | 分布式环境，集群部署    | 低延迟网络传输      |
| NPU   | `AcceleratorNPU`  | AI加速，特定神经网络模型 | 针对神经网络优化     |

## 工作负载类型

VectorSphere 定义了以下工作负载类型，系统会根据工作负载类型自动选择最适合的硬件加速器：

| 工作负载类型 | 常量值                      | 描述                 | 推荐加速器     |
|--------|--------------------------|--------------------|-----------|
| 低延迟    | `WorkloadLowLatency`     | 优化查询响应时间，适合实时应用    | FPGA, GPU |
| 高吞吐量   | `WorkloadHighThroughput` | 优化批量处理能力，适合后台任务    | GPU       |
| 分布式    | `WorkloadDistributed`    | 优化分布式环境下的性能，适合集群部署 | RDMA, GPU |
| 持久化    | `WorkloadPersistent`     | 优化持久化存储访问，适合大规模数据  | PMem      |
| 平衡     | `WorkloadBalanced`       | 平衡延迟和吞吐量，适合一般应用    | CPU, GPU  |
| 内存优化   | `WorkloadMemoryOptimal`  | 优化内存使用，适合内存受限环境    | PMem, CPU |

## 配置硬件加速器

### 1. 创建硬件管理器

```go
// 使用默认配置创建硬件管理器
hardwareManager := acceler.NewHardwareManager()

// 或者使用自定义配置创建硬件管理器
hardwareConfig := &acceler.HardwareConfig{
    CPU: acceler.CPUConfig{
        Enable:    true,
        NumCores:  0, // 0表示使用所有可用核心
        UseAVX2:   true,
        UseAVX512: true,
    },
    GPU: acceler.GPUConfig{
        Enable:      true,
        DeviceID:    0,
        MemoryLimit: 1024 * 1024 * 1024, // 1GB
        BatchSize:   1000,
    },
    FPGA: acceler.FPGAConfig{
        Enable:   false,
        DeviceID: 0,
    },
    PMem: acceler.PMemConfig{
        Enable:     false,
        MountPoint: "/mnt/pmem",
        Size:       8 * 1024 * 1024 * 1024, // 8GB
    },
    RDMA: acceler.RDMAConfig{
        Enable:    false,
        Interface: "eth0",
        Port:      18515,
    },
}

hardwareManager := acceler.NewHardwareManagerWithConfig(hardwareConfig)
```

### 2. 应用硬件管理器到向量数据库

```go
db, err := vector.NewVectorDBWithDimension(128, "cosine")
if err != nil {
    // 处理错误
}

// 应用硬件管理器
if err := db.ApplyHardwareManager(hardwareManager); err != nil {
    // 处理错误
}
```

## 使用硬件加速搜索

### 1. 基本搜索

```go
// 创建搜索选项
options := entity.SearchOptions{
    K: 10, // 返回前10个最相似的结果
}

// 执行搜索
results, err := db.OptimizedSearch(query, options.K, options)
if err != nil {
    // 处理错误
}
```

### 2. 指定硬件加速器

```go
// 使用GPU加速
options := entity.SearchOptions{
    K:      10,
    UseGPU: true,
}

// 或使用FPGA加速
options := entity.SearchOptions{
    K:       10,
    UseFPGA: true,
}
```

### 3. 根据工作负载类型自动选择加速器

```go
// 低延迟工作负载
options := entity.SearchOptions{
    K:             10,
    SearchTimeout: 5 * time.Millisecond, // 设置超时时间会自动选择低延迟工作负载
}

// 高吞吐量工作负载
options := entity.SearchOptions{
    K:         10,
    BatchSize: 200, // 设置批处理大小会自动选择高吞吐量工作负载
}

// 分布式工作负载
options := entity.SearchOptions{
    K:                10,
    DistributedSearch: true, // 启用分布式搜索
    ParallelSearch:    true, // 启用并行搜索
}

// 内存优化工作负载
options := entity.SearchOptions{
    K:               10,
    MemoryOptimized: true, // 启用内存优化
}

// 持久化存储工作负载
options := entity.SearchOptions{
    K:                 10,
    PersistentStorage: true, // 启用持久化存储
}
```

## 性能监控与调优

### 1. 获取性能指标

```go
// 获取所有加速器的性能指标
metrics := hardwareManager.GetAllPerformanceMetrics()
for accType, metric := range metrics {
    fmt.Printf("加速器 %s 性能指标:\n", accType)
    fmt.Printf("  当前延迟: %v\n", metric.LatencyCurrent)
    fmt.Printf("  当前吞吐量: %.2f ops/sec\n", metric.ThroughputCurrent)
    fmt.Printf("  资源利用率:\n")
    for resource, utilization := range metric.ResourceUtilization {
        fmt.Printf("    %s: %.2f%%\n", resource, utilization*100)
    }
}
```

### 2. 获取统计信息

```go
// 获取所有加速器的统计信息
stats := hardwareManager.GetAllStats()
for accType, stat := range stats {
    fmt.Printf("加速器 %s 统计信息:\n", accType)
    fmt.Printf("  总操作数: %d\n", stat.TotalOperations)
    fmt.Printf("  成功操作数: %d\n", stat.SuccessfulOps)
    fmt.Printf("  失败操作数: %d\n", stat.FailedOps)
    fmt.Printf("  平均延迟: %v\n", stat.AverageLatency)
    fmt.Printf("  吞吐量: %.2f ops/sec\n", stat.Throughput)
    fmt.Printf("  内存利用率: %.2f%%\n", stat.MemoryUtilization*100)
    fmt.Printf("  错误率: %.2f%%\n", stat.ErrorRate*100)
}
```

### 3. 自动调优

```go
// 启用自动调优
db.EnableAutoTuning(true)

// 设置自动调优参数
tuningOptions := entity.TuningOptions{
    TuningInterval:    time.Minute * 10, // 每10分钟调优一次
    PerformanceTarget: entity.PerformanceTargetLatency, // 以延迟为优化目标
    MaxMemoryUsage:    8 * 1024 * 1024 * 1024, // 最大内存使用量 8GB
}
db.SetAutoTuningOptions(tuningOptions)
```

## 最佳实践

### 1. 选择合适的工作负载类型

- **实时查询应用**：使用低延迟工作负载（`WorkloadLowLatency`）
- **批量处理任务**：使用高吞吐量工作负载（`WorkloadHighThroughput`）
- **集群部署环境**：使用分布式工作负载（`WorkloadDistributed`）
- **大规模数据集**：使用持久化工作负载（`WorkloadPersistent`）
- **内存受限环境**：使用内存优化工作负载（`WorkloadMemoryOptimal`）
- **一般应用场景**：使用平衡工作负载（`WorkloadBalanced`）

### 2. 硬件加速器选择建议

- **GPU**：适合大规模并行计算，高维向量，批量处理
- **FPGA**：适合低延迟要求，特定算法优化
- **PMem**：适合大规模数据集，内存受限环境
- **RDMA**：适合分布式环境，集群部署

### 3. 性能优化建议

- 对于频繁查询的场景，启用结果缓存：`options.EnableResultCache = true`
- 对于大规模数据集，考虑使用多阶段搜索：`options.EnableMultiStage = true`
- 对于高维向量，考虑使用PQ压缩：`options.UsePQCompression = true`
- 对于批量查询，设置合适的批处理大小：`options.BatchSize = 100`
- 定期监控硬件加速器的性能指标和统计信息，及时调整配置

## 故障排除

### 1. GPU 加速器不可用

- 检查 GPU 驱动是否正确安装
- 检查 CUDA 库是否正确安装
- 检查 GPU 内存是否足够
- 检查 GPU 健康状态：`hardwareManager.CheckGPUHealth()`

### 2. 搜索性能不佳

- 检查是否选择了合适的工作负载类型
- 检查是否启用了合适的硬件加速器
- 检查数据规模是否超出了硬件加速器的处理能力
- 检查是否有其他进程占用了硬件资源

### 3. 硬件加速器错误

- 检查硬件加速器的错误日志
- 检查硬件加速器的统计信息，特别是错误率
- 尝试重新初始化硬件加速器：`hardwareManager.ReInitialize(acceler.AcceleratorGPU)`
- 如果错误持续，尝试禁用该硬件加速器：`hardwareManager.Disable(acceler.AcceleratorGPU)`

## 结论

通过合理配置和使用硬件加速器，VectorSphere 可以显著提高向量搜索的性能。根据不同的应用场景和工作负载类型，选择合适的硬件加速策略，可以获得最佳的性能体验。