package acceler

import (
	"fmt"
	"sync"
	"time"
)

// BaseAccelerator 基础加速器结构，包含所有加速器的通用字段和方法
type BaseAccelerator struct {
	deviceID        int
	initialized     bool
	available       bool
	mu              sync.RWMutex
	indexType       string
	dimension       int
	strategy        *ComputeStrategySelector
	currentStrategy ComputeStrategy
	dataSize        int
	hardwareManager *HardwareManager
	lastStatsTime   time.Time
	startTime       time.Time

	// 统计信息
	stats HardwareStats

	// 性能指标
	performanceMetrics PerformanceMetrics

	// 能力信息
	capabilities HardwareCapabilities
}

// NewBaseAccelerator 创建基础加速器实例
func NewBaseAccelerator(deviceID int, indexType string, capabilities HardwareCapabilities, stats HardwareStats) *BaseAccelerator {
	return &BaseAccelerator{
		deviceID:      deviceID,
		indexType:     indexType,
		initialized:   false,
		available:     true,
		strategy:      NewComputeStrategySelector(),
		lastStatsTime: time.Now(),
		startTime:     time.Now(),
		capabilities:  capabilities,
		stats:         stats,
	}
}

// IsAvailable 检查加速器是否可用
func (b *BaseAccelerator) IsAvailable() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.available
}

// IsInitialized 检查加速器是否已初始化
func (b *BaseAccelerator) IsInitialized() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.initialized
}

// SetHardwareManager 设置硬件管理器
func (b *BaseAccelerator) SetHardwareManager(hm *HardwareManager) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.hardwareManager = hm
	if b.strategy != nil {
		b.strategy.SetHardwareManager(hm)
	}
}

// GetCapabilities 获取硬件能力
func (b *BaseAccelerator) GetCapabilities() HardwareCapabilities {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.capabilities
}

// GetStats 获取统计信息
func (b *BaseAccelerator) GetStats() HardwareStats {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.stats
}

// GetPerformanceMetrics 获取性能指标
func (b *BaseAccelerator) GetPerformanceMetrics() PerformanceMetrics {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.performanceMetrics
}

// UpdateStats 更新统计信息
func (b *BaseAccelerator) UpdateStats(duration time.Duration, operations int64, success bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.stats.TotalOperations += operations
	if success {
		b.stats.SuccessfulOps += operations
	} else {
		b.stats.FailedOps += operations
	}

	// 更新平均延迟
	if b.stats.TotalOperations > 0 {
		totalTime := time.Duration(b.stats.TotalOperations) * b.stats.AverageLatency
		totalTime += duration
		b.stats.AverageLatency = totalTime / time.Duration(b.stats.TotalOperations)
	} else {
		b.stats.AverageLatency = duration
	}

	// 更新错误率
	if b.stats.TotalOperations > 0 {
		b.stats.ErrorRate = float64(b.stats.FailedOps) / float64(b.stats.TotalOperations)
	}

	b.stats.LastUsed = time.Now()
}

// UpdatePerformanceMetrics 更新性能指标
func (b *BaseAccelerator) UpdatePerformanceMetrics(latency time.Duration, throughput float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.performanceMetrics.LatencyCurrent = latency
	b.performanceMetrics.ThroughputCurrent = throughput

	// 更新最小/最大延迟
	if b.performanceMetrics.LatencyMin == 0 || latency < b.performanceMetrics.LatencyMin {
		b.performanceMetrics.LatencyMin = latency
	}
	if latency > b.performanceMetrics.LatencyMax {
		b.performanceMetrics.LatencyMax = latency
	}

	// 更新峰值吞吐量
	if throughput > b.performanceMetrics.ThroughputPeak {
		b.performanceMetrics.ThroughputPeak = throughput
	}
}

// SetAvailable 设置可用状态
func (b *BaseAccelerator) SetAvailable(available bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.available = available
}

// SetInitialized 设置初始化状态
func (b *BaseAccelerator) SetInitialized(initialized bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.initialized = initialized
}

// ValidateInputs 验证输入参数
func (b *BaseAccelerator) ValidateInputs(query []float64, vectors [][]float64) error {
	if !b.IsInitialized() {
		return fmt.Errorf("加速器未初始化")
	}

	if !b.IsAvailable() {
		return fmt.Errorf("加速器不可用")
	}

	if len(query) == 0 {
		return fmt.Errorf("查询向量为空")
	}

	if len(vectors) == 0 {
		return fmt.Errorf("目标向量集为空")
	}

	// 检查维度一致性
	for i, vector := range vectors {
		if len(vector) != len(query) {
			return fmt.Errorf("向量 %d 维度不匹配: 期望 %d, 实际 %d", i, len(query), len(vector))
		}
	}

	return nil
}

// ValidateBatchInputs 验证批量输入参数
func (b *BaseAccelerator) ValidateBatchInputs(queries [][]float64, vectors [][]float64) error {
	if !b.IsInitialized() {
		return fmt.Errorf("加速器未初始化")
	}

	if !b.IsAvailable() {
		return fmt.Errorf("加速器不可用")
	}

	if len(queries) == 0 {
		return fmt.Errorf("查询向量集为空")
	}

	if len(vectors) == 0 {
		return fmt.Errorf("目标向量集为空")
	}

	// 检查查询向量维度一致性
	if len(queries) > 0 {
		queryDim := len(queries[0])
		for i, query := range queries {
			if len(query) != queryDim {
				return fmt.Errorf("查询向量 %d 维度不匹配: 期望 %d, 实际 %d", i, queryDim, len(query))
			}
		}

		// 检查目标向量维度一致性
		for i, vector := range vectors {
			if len(vector) != queryDim {
				return fmt.Errorf("目标向量 %d 维度不匹配: 期望 %d, 实际 %d", i, queryDim, len(vector))
			}
		}
	}

	return nil
}

// AutoTune 自动调优（基础实现）
func (b *BaseAccelerator) AutoTune(workload WorkloadProfile) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// 基础自动调优逻辑
	if b.strategy != nil {
		b.currentStrategy = b.strategy.SelectOptimalStrategy(int(workload.DataSize), workload.VectorDimension)
	}

	return nil
}

// GetDeviceID 获取设备ID
func (b *BaseAccelerator) GetDeviceID() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.deviceID
}

// GetIndexType 获取索引类型
func (b *BaseAccelerator) GetIndexType() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.indexType
}

// GetDimension 获取向量维度
func (b *BaseAccelerator) GetDimension() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.dimension
}

// SetDimension 设置向量维度
func (b *BaseAccelerator) SetDimension(dimension int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.dimension = dimension
}
