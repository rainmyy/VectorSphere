# VectorSphere 硬件加速器库

## 概述

本库提供了统一的硬件加速接口，整合了多种硬件加速技术，为向量数据库提供高性能计算能力。支持CPU、GPU、FPGA、持久内存(PMem)和RDMA网络加速。

## 架构设计

### 核心组件

1. **统一加速器接口** (`UnifiedAccelerator`)
   - 定义了所有硬件加速器的通用接口
   - 支持生命周期管理、核心计算、高级功能等

2. **硬件管理器** (`HardwareManager`)
   - 管理所有注册的硬件加速器
   - 提供加速器选择和负载均衡
   - 系统健康监控和性能统计

3. **向量数据库适配器** (`VectorDBAdapter`)
   - 为向量数据库提供统一的硬件加速接口
   - 自动选择最佳加速器
   - 工作负载优化和性能调优

### 支持的加速器类型

#### 1. CPU加速器 (`CPUAccelerator`)
- **特性**: SIMD指令优化、多线程并行
- **适用场景**: 通用计算、小规模数据集
- **优势**: 兼容性好、延迟低

#### 2. GPU加速器 (`GPUAccelerator`)
- **特性**: CUDA/OpenCL支持、大规模并行计算
- **适用场景**: 大规模向量计算、高吞吐量需求
- **优势**: 计算能力强、适合批量处理

#### 3. FPGA加速器 (`FPGAAccelerator`)
- **特性**: 可重配置硬件、超低延迟
- **适用场景**: 实时系统、特定算法优化
- **优势**: 延迟极低、功耗可控

#### 4. 持久内存加速器 (`PMemAccelerator`)
- **特性**: 非易失性内存、大容量存储
- **适用场景**: 大规模数据集、持久化需求
- **优势**: 容量大、数据持久化

#### 5. RDMA网络加速器 (`RDMAAccelerator`)
- **特性**: 远程直接内存访问、分布式计算
- **适用场景**: 分布式向量搜索、集群计算
- **优势**: 网络延迟低、扩展性好

## 使用指南

### 基本使用

```go
package main

import (
    "VectorSphere/src/library/acceler"
    "fmt"
)

func main() {
    // 创建向量数据库适配器
    adapter := acceler.NewVectorDBAdapter()
    defer adapter.Shutdown()
    
    // 检查硬件加速状态
    if adapter.IsEnabled() {
        fmt.Println("硬件加速已启用")
    }
    
    // 生成测试数据
    query := []float64{1.0, 2.0, 3.0, 4.0}
    database := [][]float64{
        {1.1, 2.1, 3.1, 4.1},
        {0.9, 1.9, 2.9, 3.9},
        {1.2, 2.2, 3.2, 4.2},
    }
    
    // 执行向量搜索
    results, err := adapter.Search(query, database, 2, acceler.WorkloadBalanced)
    if err != nil {
        fmt.Printf("搜索失败: %v\n", err)
        return
    }
    
    // 显示结果
    for i, result := range results {
        fmt.Printf("结果 %d: ID=%s, 相似度=%.6f\n", i+1, result.ID, result.Similarity)
    }
}
```

### 工作负载优化

```go
// 为不同工作负载类型优化配置
workloadTypes := []string{
    acceler.WorkloadLowLatency,    // 低延迟优化
    acceler.WorkloadHighThroughput, // 高吞吐量优化
    acceler.WorkloadDistributed,   // 分布式优化
    acceler.WorkloadPersistent,    // 持久化优化
    acceler.WorkloadBalanced,      // 平衡优化
}

for _, workloadType := range workloadTypes {
    err := adapter.OptimizeForWorkload(workloadType, queryCount, dimension)
    if err != nil {
        fmt.Printf("优化失败: %v\n", err)
        continue
    }
    
    // 获取推荐的加速器
    workload := acceler.CreateWorkloadProfile(workloadType, queryCount, dimension)
    recommended := adapter.GetRecommendedAccelerator(workload)
    fmt.Printf("推荐加速器: %s\n", recommended)
}
```

### 硬件管理器使用

```go
// 获取硬件管理器
hm := adapter.GetHardwareManager()

// 获取可用加速器
available := hm.GetAvailableAccelerators()
fmt.Printf("可用加速器: %v\n", available)

// 获取系统信息
systemInfo := hm.GetSystemInfo()
fmt.Printf("系统信息: %+v\n", systemInfo)

// 健康检查
health := hm.HealthCheck()
fmt.Printf("健康状态: %+v\n", health)

// 获取性能统计
stats := hm.GetAllStats()
for name, stat := range stats {
    fmt.Printf("%s: 总操作=%d, 成功率=%.2f%%\n", 
        name, stat.TotalOperations, 
        float64(stat.SuccessfulOps)/float64(stat.TotalOperations)*100)
}
```

## 配置说明

### 硬件加速配置

每种加速器都有专门的配置结构：

- `CPUConfig`: CPU优化配置
- `GPUConfig`: GPU设备和CUDA配置
- `FPGAConfig`: FPGA并行和优化配置
- `PMemConfig`: 持久内存命名空间和性能配置
- `RDMAConfig`: RDMA网络和设备配置

### 性能调优参数

- **BatchSize**: 批处理大小，影响吞吐量和内存使用
- **ThreadCount**: 线程数量，影响并行度
- **CacheSize**: 缓存大小，影响内存访问性能
- **Timeout**: 超时设置，影响响应时间

## 性能特性

### 延迟对比

| 加速器类型 | 典型延迟 | 适用场景 |
|-----------|----------|----------|
| CPU       | 1-10ms   | 小规模、实时 |
| GPU       | 5-50ms   | 大规模、批处理 |
| FPGA      | 0.1-1ms  | 超低延迟 |
| PMem      | 1-5ms    | 大数据集 |
| RDMA      | 1-10ms   | 分布式 |

### 吞吐量对比

| 加速器类型 | 典型吞吐量 | 扩展性 |特性|
|-----------|------------|--------|--------|
| CPU       | 1K-10K ops/s | 中等 |CPU加速器，支持SIMD优化和多线|
| GPU       | 10K-100K ops/s | 高 |GPU加速器，整合FAISS-GPU和CUDA支持|
| FPGA      | 5K-50K ops/s | 中等 |FPGA加速器，提供超低延迟计算能力|
| PMem      | 1K-20K ops/s | 高 |持久内存加速器，支持大容量数据处理|
| RDMA      | 5K-100K ops/s | 极高 |RDMA网络加速器，支持分布式计算|

## 最佳实践

### 1. 加速器选择

- **低延迟需求**: 优先选择FPGA或CPU
- **高吞吐量需求**: 优先选择GPU或RDMA
- **大数据集**: 优先选择PMem或GPU
- **分布式场景**: 优先选择RDMA

### 2. 性能优化

- 根据数据规模调整批处理大小
- 合理设置线程数量避免过度竞争
- 启用缓存机制提高重复查询性能
- 监控硬件资源使用情况

### 3. 错误处理

- 实现加速器故障转移机制
- 监控硬件健康状态
- 记录性能指标用于调优
- 设置合理的超时时间

## 扩展开发

### 添加新的加速器

1. 实现 `UnifiedAccelerator` 接口
2. 在 `HardwareManager` 中注册新加速器
3. 添加相应的配置结构
4. 实现硬件检测逻辑

### 自定义工作负载

1. 定义新的工作负载类型常量
2. 在 `CreateWorkloadProfile` 中添加处理逻辑
3. 更新加速器选择算法
4. 添加相应的性能调优参数

## 故障排除

### 常见问题

1. **加速器初始化失败**
   - 检查硬件驱动是否正确安装
   - 验证硬件设备是否可用
   - 查看系统日志获取详细错误信息

2. **性能不如预期**
   - 检查工作负载配置是否合适
   - 监控硬件资源使用情况
   - 调整批处理大小和线程数量

3. **内存不足**
   - 减少批处理大小
   - 启用内存优化选项
   - 考虑使用PMem扩展内存容量
   

### 智能工作负载优化
- 低延迟模式 : 优先使用FPGA和CPU加速器
- 高吞吐量模式 : 优先使用GPU和RDMA加速器
- 分布式模式 : 专门优化RDMA网络加速
- 持久化模式 : 充分利用PMem大容量特性
- 平衡模式 : 综合考虑各种因素选择最佳方案
### 调试工具

- 使用 `GetPerformanceReport()` 获取详细性能报告
- 使用 `HealthCheck()` 检查硬件状态
- 启用详细日志记录
- 使用性能分析工具监控资源使用

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本项目。请确保：

1. 代码符合项目编码规范
2. 添加适当的测试用例
3. 更新相关文档
4. 通过所有现有测试

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者
- 参与项目讨论区