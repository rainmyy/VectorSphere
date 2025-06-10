package vector

import (
	"VectorSphere/src/library/log"
	"sync"
	"time"
)

// 添加自适应索引选择器
type AdaptiveIndexSelector struct {
	db                   *VectorDB
	performanceWindow    []PerformanceRecord
	windowSize           int
	lastOptimization     time.Time
	optimizationInterval time.Duration
	mu                   sync.RWMutex
}

type PerformanceRecord struct {
	Strategy    IndexStrategy
	Latency     time.Duration
	Quality     float64
	Timestamp   time.Time
	VectorCount int
	Dimension   int
}

// NewAdaptiveIndexSelector 创建自适应索引选择器
func NewAdaptiveIndexSelector(db *VectorDB) *AdaptiveIndexSelector {
	return &AdaptiveIndexSelector{
		db:                   db,
		performanceWindow:    make([]PerformanceRecord, 0),
		windowSize:           100, // 保留最近100次查询的性能记录
		lastOptimization:     time.Now(),
		optimizationInterval: 5 * time.Minute, // 每5分钟优化一次
	}
}

// RecordPerformance 记录性能数据
func (ais *AdaptiveIndexSelector) RecordPerformance(strategy IndexStrategy, latency time.Duration, quality float64, vectorCount, dimension int) {
	ais.mu.Lock()
	defer ais.mu.Unlock()

	record := PerformanceRecord{
		Strategy:    strategy,
		Latency:     latency,
		Quality:     quality,
		Timestamp:   time.Now(),
		VectorCount: vectorCount,
		Dimension:   dimension,
	}

	ais.performanceWindow = append(ais.performanceWindow, record)

	// 保持窗口大小
	if len(ais.performanceWindow) > ais.windowSize {
		ais.performanceWindow = ais.performanceWindow[1:]
	}

	// 检查是否需要优化
	if time.Since(ais.lastOptimization) > ais.optimizationInterval {
		go ais.optimizeStrategySelection()
	}
}

// optimizeStrategySelection 优化策略选择
func (ais *AdaptiveIndexSelector) optimizeStrategySelection() {
	ais.mu.Lock()
	defer ais.mu.Unlock()

	if len(ais.performanceWindow) < 10 {
		return // 数据不足，无法优化
	}

	// 分析不同策略的性能表现
	strategyPerf := make(map[IndexStrategy][]PerformanceRecord)
	for _, record := range ais.performanceWindow {
		strategyPerf[record.Strategy] = append(strategyPerf[record.Strategy], record)
	}

	// 计算每种策略的平均性能
	for strategy, records := range strategyPerf {
		if len(records) < 3 {
			continue // 样本太少
		}

		var totalLatency time.Duration
		var totalQuality float64
		for _, record := range records {
			totalLatency += record.Latency
			totalQuality += record.Quality
		}

		avgLatency := totalLatency / time.Duration(len(records))
		avgQuality := totalQuality / float64(len(records))

		log.Info("策略 %v 性能统计: 平均延迟=%v, 平均质量=%.3f, 样本数=%d",
			strategy, avgLatency, avgQuality, len(records))
	}

	ais.lastOptimization = time.Now()
}
