# FPGA加速器系统

本文档介绍VectorSphere项目中的FPGA加速器系统，该系统提供了硬件加速的向量计算能力，并在不支持FPGA的环境下自动降级到软件模拟。

## 系统架构

### 核心组件

1. **fpga_accelerator.go** - 真实FPGA硬件加速器实现
2. **fpga_accelerator_stub.go** - FPGA模拟器实现
3. **fpga_factory.go** - 工厂模式，自动选择真实FPGA或模拟器
4. **fpga_example.go** - 使用示例和测试代码

### 接口统一性

两个实现都遵循`UnifiedAccelerator`接口，确保API完全一致：

```go
type UnifiedAccelerator interface {
    GetType() string
    IsAvailable() bool
    Initialize() error
    Shutdown() error
    ComputeDistance(query []float64, vectors [][]float64) ([]float64, error)
    BatchComputeDistance(queries [][]float64, vectors [][]float64) ([][]float64, error)
    BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error)
    // ... 更多方法
}
```

## 自动选择机制

系统通过`FPGAFactory`自动选择使用真实FPGA还是模拟器：

### 选择逻辑

1. **环境变量控制**：
   ```bash
   export FPGA_USE_STUB=true  # 强制使用模拟器
   export FPGA_USE_STUB=false # 尝试使用真实FPGA
   ```

2. **测试环境检测**：
   - 自动检测是否在运行测试
   - 测试环境下默认使用模拟器

3. **操作系统兼容性**：
   - Linux：支持真实FPGA
   - Windows：检查FPGA驱动是否存在
   - macOS：默认使用模拟器

4. **硬件可用性检测**：
   - 尝试初始化真实FPGA
   - 失败时自动降级到模拟器

## 使用方法

### 基本使用

```go
package main

import (
    "log"
    "github.com/VectorSphere/src/library/acceler"
)

func main() {
    // 创建工厂
    factory := acceler.NewFPGAFactory()
    
    // 获取推荐配置
    config := factory.GetRecommendedConfig()
    
    // 创建加速器（自动选择FPGA或模拟器）
    accelerator, err := factory.CreateFPGAAccelerator(config)
    if err != nil {
        log.Fatal(err)
    }
    defer accelerator.Shutdown()
    
    // 初始化
    if err := accelerator.Initialize(); err != nil {
        log.Fatal(err)
    }
    
    // 使用加速器进行计算
    query := []float64{1.0, 2.0, 3.0, 4.0}
    vectors := [][]float64{
        {1.1, 2.1, 3.1, 4.1},
        {2.0, 3.0, 4.0, 5.0},
    }
    
    distances, err := accelerator.ComputeDistance(query, vectors)
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("计算结果: %v", distances)
    log.Printf("使用的加速器类型: %s", accelerator.GetType())
}
```

### 完整示例

```go
// 运行完整示例
example := acceler.NewFPGAExample()
if err := example.RunCompleteExample(); err != nil {
    log.Fatal(err)
}
```

## 配置选项

### FPGAConfig结构

```go
type FPGAConfig struct {
    DeviceID       int                        // FPGA设备ID
    ClockFrequency int                        // 时钟频率(MHz)
    BufferSize     int                        // 缓冲区大小
    PowerLimit     float64                    // 功耗限制(W)
    PipelineDepth  int                        // 流水线深度
    Parallelism    FPGAParallelismConfig      // 并行配置
    Optimization   FPGAOptimizationConfig     // 优化配置
    Reconfiguration FPGAReconfigurationConfig // 重配置选项
}
```

### 推荐配置

```go
config := &FPGAConfig{
    DeviceID:       0,
    ClockFrequency: 200,        // 200MHz
    BufferSize:     1024 * 1024, // 1MB
    PowerLimit:     25.0,       // 25W
    PipelineDepth:  8,
    Parallelism: FPGAParallelismConfig{
        ComputeUnits:    4,
        EnableStreaming: true,
    },
    Optimization: FPGAOptimizationConfig{
        EnableBRAM:    true,
        EnableDSP:     true,
        MemoryBanking: 4,
    },
}
```

## 功能特性

### 向量计算

- **距离计算**：欧几里得距离、余弦相似度
- **批量计算**：支持批量向量操作
- **并行处理**：利用FPGA并行计算能力

### 搜索加速

- **向量搜索**：高效的最近邻搜索
- **结果优化**：FPGA硬件级别的结果排序和过滤
- **流水线处理**：重叠计算和数据传输

### 内存优化

- **布局优化**：AOS、SOA、分块、混合布局
- **数据预取**：智能数据预取机制
- **缓存优化**：BRAM、URAM利用

### 性能监控

- **实时统计**：操作计数、延迟、吞吐量
- **硬件监控**：温度、功耗、利用率
- **性能分析**：延迟分布、性能指标

## 性能对比

### 真实FPGA vs 模拟器

| 功能 | 真实FPGA | 模拟器 |
|------|----------|--------|
| 计算性能 | 硬件加速，高并行 | CPU模拟，较慢 |
| 延迟 | 微秒级 | 毫秒级 |
| 功耗 | 优化的硬件功耗 | CPU功耗 |
| 可用性 | 需要硬件支持 | 任何环境 |
| 开发测试 | 需要真实硬件 | 便于开发调试 |

### 性能优化建议

1. **生产环境**：使用真实FPGA获得最佳性能
2. **开发环境**：使用模拟器进行快速开发
3. **测试环境**：使用模拟器确保测试稳定性
4. **混合部署**：根据负载动态选择

## 故障排除

### 常见问题

1. **FPGA初始化失败**
   ```
   错误：FPGA设备初始化失败
   解决：检查FPGA驱动和硬件连接，系统会自动降级到模拟器
   ```

2. **比特流加载失败**
   ```
   错误：加载比特流失败
   解决：检查比特流文件路径和格式
   ```

3. **性能不如预期**
   ```
   解决：调整并行度、时钟频率、内存配置
   ```

### 调试模式

```bash
# 启用详细日志
export FPGA_DEBUG=true

# 强制使用模拟器进行调试
export FPGA_USE_STUB=true

# 启用性能分析
export FPGA_PROFILE=true
```

## 扩展开发

### 添加新的加速操作

1. 在`UnifiedAccelerator`接口中添加方法
2. 在`fpga_accelerator.go`中实现硬件版本
3. 在`fpga_accelerator_stub.go`中实现模拟版本
4. 更新测试和示例

### 自定义配置

```go
// 创建自定义配置
config := &FPGAConfig{
    // 自定义参数
}

// 验证配置
if err := factory.ValidateConfig(config); err != nil {
    log.Fatal(err)
}

// 使用自定义配置创建加速器
accelerator, err := factory.CreateFPGAAccelerator(config)
```

## 最佳实践

1. **资源管理**：及时调用`Shutdown()`释放资源
2. **错误处理**：检查所有返回的错误
3. **性能监控**：定期检查性能统计
4. **配置优化**：根据工作负载调整配置
5. **测试覆盖**：同时测试FPGA和模拟器版本

## 许可证

本项目遵循MIT许可证。详见LICENSE文件。