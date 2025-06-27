# VectorSphere 完整硬件加速器管理系统指南

## 概述

VectorSphere 的硬件加速器管理系统提供了一个完整的解决方案，用于管理、监控和自动恢复各种硬件加速器（GPU、RDMA等）。该系统包含错误处理、健康监控、自动恢复等高级功能。

## 系统架构

### 核心组件

1. **HardwareManager** - 硬件管理器核心
2. **ErrorHandler** - 统一错误处理器
3. **HealthMonitor** - 健康监控器
4. **RecoveryManager** - 自动恢复管理器

### 组件关系图

```
┌─────────────────┐
│ HardwareManager │
├─────────────────┤
│ - accelerators  │
│ - errorHandler  │
│ - healthMonitor │
│ - recoveryMgr   │
└─────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──┐
│ GPU   │ │RDMA │
│Accel  │ │Accel│
└───────┘ └─────┘
```

## 功能特性

### 1. 错误处理系统

#### 错误分类
- **初始化错误** - 加速器初始化失败
- **运行时错误** - 执行过程中的错误
- **内存错误** - 内存分配或访问错误
- **网络错误** - RDMA等网络相关错误
- **硬件错误** - 硬件故障
- **配置错误** - 配置参数错误
- **超时错误** - 操作超时
- **未知错误** - 其他未分类错误

#### 错误处理功能
- 错误分类和统计
- 堆栈跟踪记录
- 重试机制
- 全局错误处理
- 安全调用包装

### 2. 健康监控系统

#### 监控指标
- **可用性检查** - 加速器是否可用
- **初始化状态** - 是否已正确初始化
- **错误率** - 操作失败率
- **响应时间** - 操作响应延迟
- **内存使用率** - 内存占用情况
- **CPU使用率** - CPU占用情况
- **吞吐量** - 处理能力

#### 健康状态
- **健康** - 所有指标正常
- **警告** - 部分指标异常但仍可用
- **严重** - 关键指标异常，需要干预
- **未知** - 无法获取状态信息

### 3. 自动恢复系统

#### 恢复策略
- **重新初始化** - 重新初始化加速器
- **重启** - 完全重启加速器
- **回退** - 切换到其他可用加速器
- **禁用** - 禁用故障加速器

#### 恢复流程
1. 健康监控检测到异常
2. 确定恢复策略
3. 执行恢复操作
4. 验证恢复结果
5. 记录恢复历史

## 使用指南

### 基本使用

```go
package main

import (
    "github.com/VectorSphere/src/library/acceler"
)

func main() {
    // 1. 创建硬件管理器
    config := acceler.GetDefaultHardwareConfig()
    hm := acceler.NewHardwareManagerWithConfig(config)
    
    // 2. 启动监控和恢复
    hm.StartHealthMonitoring()
    hm.StartRecoveryManager()
    
    // 3. 使用加速器
    gpuAccel := hm.GetGPUAccelerator()
    if gpuAccel != nil {
        // 安全调用
        results, err := hm.SafeGPUBatchSearch(query, database, k)
        if err != nil {
            // 处理错误
        }
    }
    
    // 4. 清理资源
    defer hm.StopHealthMonitoring()
    defer hm.StopRecoveryManager()
}
```

### 高级配置

#### 错误处理配置

```go
// 自定义错误处理
errorHandler := acceler.NewErrorHandler()
errorHandler.SetMaxRetries(5)
errorHandler.SetRetryDelay(time.Second * 2)
```

#### 健康监控配置

```go
// 设置监控间隔
healthMonitor := acceler.NewHealthMonitor(hm)
healthMonitor.SetMonitorInterval(30 * time.Second)
healthMonitor.Start()
```

#### 恢复管理配置

```go
// 自定义恢复配置
recoveryConfig := &acceler.RecoveryConfig{
    MaxRetries:          3,
    RetryInterval:       30 * time.Second,
    HealthCheckInterval: 60 * time.Second,
    AutoRecoveryEnabled: true,
    FallbackEnabled:     true,
}

recoveryManager := acceler.NewRecoveryManager(hm, recoveryConfig)
recoveryManager.Start()
```

## API 参考

### HardwareManager 主要方法

#### 基础管理
- `GetGPUAccelerator() UnifiedAccelerator` - 获取GPU加速器
- `RegisterGPUAccelerator(acc UnifiedAccelerator) error` - 注册GPU加速器
- `GetAvailableAccelerators() []string` - 获取可用加速器列表
- `SetDefaultAccelerator(acc UnifiedAccelerator)` - 设置默认加速器

#### 安全调用
- `SafeGPUCall(operation string, fn func() error) error` - 安全GPU调用
- `SafeGPUBatchSearch(query []float64, database [][]float64, k int) ([]AccelResult, error)` - 安全GPU批量搜索

#### 错误管理
- `GetErrorStats() map[string]int` - 获取错误统计
- `GetLastError(acceleratorType, operation string) *AcceleratorError` - 获取最后错误
- `ResetErrorCount(acceleratorType, operation string)` - 重置错误计数

#### 健康监控
- `StartHealthMonitoring()` - 启动健康监控
- `StopHealthMonitoring()` - 停止健康监控
- `GetHealthReport(acceleratorType string) *HealthReport` - 获取健康报告
- `GetAllHealthReports() map[string]*HealthReport` - 获取所有健康报告
- `GetOverallHealth() HealthStatus` - 获取整体健康状态
- `IsHealthy(acceleratorType string) bool` - 检查是否健康

#### 恢复管理
- `StartRecoveryManager()` - 启动恢复管理
- `StopRecoveryManager()` - 停止恢复管理
- `GetRecoveryHistory() []RecoveryAction` - 获取恢复历史
- `GetRetryCount(acceleratorType string) int` - 获取重试次数
- `ResetRetryCount(acceleratorType string)` - 重置重试次数
- `UpdateRecoveryConfig(config *RecoveryConfig)` - 更新恢复配置

### 数据结构

#### HealthReport
```go
type HealthReport struct {
    AcceleratorType string        `json:"accelerator_type"`
    Status          HealthStatus  `json:"status"`
    Message         string        `json:"message"`
    Timestamp       time.Time     `json:"timestamp"`
    Metrics         HealthMetrics `json:"metrics"`
}
```

#### HealthMetrics
```go
type HealthMetrics struct {
    IsAvailable     bool          `json:"is_available"`
    IsInitialized   bool          `json:"is_initialized"`
    ErrorRate       float64       `json:"error_rate"`
    LastErrorTime   *time.Time    `json:"last_error_time,omitempty"`
    ResponseTime    time.Duration `json:"response_time"`
    MemoryUsage     float64       `json:"memory_usage"`
    CPUUsage        float64       `json:"cpu_usage"`
    Throughput      float64       `json:"throughput"`
}
```

#### RecoveryAction
```go
type RecoveryAction struct {
    AcceleratorType string           `json:"accelerator_type"`
    Strategy        RecoveryStrategy `json:"strategy"`
    Timestamp       time.Time        `json:"timestamp"`
    Reason          string           `json:"reason"`
    Success         bool             `json:"success"`
    ErrorMessage    string           `json:"error_message,omitempty"`
}
```

## 最佳实践

### 1. 初始化顺序
1. 创建硬件管理器
2. 启动健康监控
3. 启动恢复管理
4. 注册自定义加速器（如需要）

### 2. 错误处理
- 始终使用安全调用方法
- 定期检查错误统计
- 根据错误类型采取相应措施

### 3. 性能优化
- 合理设置监控间隔
- 避免频繁的健康检查
- 使用批量操作提高效率

### 4. 资源管理
- 及时停止监控和恢复服务
- 正确关闭加速器资源
- 定期清理历史记录

## 故障排除

### 常见问题

#### 1. GPU加速器初始化失败
**症状**: `GetGPUAccelerator()` 返回 `nil`
**解决方案**:
- 检查GPU驱动是否正确安装
- 验证CUDA环境配置
- 查看错误日志获取详细信息

#### 2. 健康监控显示异常
**症状**: 健康状态为警告或严重
**解决方案**:
- 检查具体的健康指标
- 查看错误统计和最后错误
- 考虑手动重启加速器

#### 3. 自动恢复失败
**症状**: 恢复历史显示失败记录
**解决方案**:
- 检查恢复策略是否合适
- 验证硬件状态
- 考虑禁用故障加速器

### 调试技巧

1. **启用详细日志**
```go
log.SetLevel(log.DebugLevel)
```

2. **监控关键指标**
```go
// 定期检查健康状态
go func() {
    ticker := time.NewTicker(10 * time.Second)
    for range ticker.C {
        status := hm.GetOverallHealth()
        log.Printf("整体健康状态: %s", status)
    }
}()
```

3. **错误统计分析**
```go
// 分析错误模式
errorStats := hm.GetErrorStats()
for key, count := range errorStats {
    if count > threshold {
        log.Printf("高频错误: %s (%d次)", key, count)
    }
}
```

## 扩展开发

### 自定义加速器

实现 `UnifiedAccelerator` 接口:

```go
type CustomAccelerator struct {
    // 自定义字段
}

func (ca *CustomAccelerator) Initialize() error {
    // 初始化逻辑
}

func (ca *CustomAccelerator) IsAvailable() bool {
    // 可用性检查
}

// 实现其他接口方法...
```

### 自定义恢复策略

扩展 `RecoveryStrategy` 枚举并实现相应逻辑:

```go
const (
    RecoveryStrategyCustom RecoveryStrategy = iota + 100
)

func (rm *RecoveryManager) executeCustomRecovery(acceleratorType string) error {
    // 自定义恢复逻辑
}
```

## 版本历史

- **v1.0.0** - 基础硬件管理功能
- **v1.1.0** - 添加错误处理系统
- **v1.2.0** - 添加健康监控系统
- **v1.3.0** - 添加自动恢复系统
- **v1.4.0** - 完整的管理系统集成

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。