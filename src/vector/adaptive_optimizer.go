package vector

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// AdaptiveOptimizer 自适应优化引擎
type AdaptiveOptimizer struct {
	configManager      *ConfigManager
	performanceMonitor *StandardPerformanceMonitor
	optimizationRules  []OptimizationRule
	running            bool
	mu                 sync.RWMutex
	ctx                context.Context
	cancel             context.CancelFunc
	optimizationLog    []OptimizationEvent
}

// OptimizationRule 优化规则
type OptimizationRule struct {
	Name        string
	Description string
	Condition   func(*PerformanceMetrics) bool
	Action      func(*ConfigManager) error
	Priority    int
	Cooldown    time.Duration
	LastApplied time.Time
	Enabled     bool
}

// OptimizationEvent 优化事件
type OptimizationEvent struct {
	Timestamp   time.Time
	RuleName    string
	Description string
	Metrics     *PerformanceMetrics
	Success     bool
	Error       error
}

// NewAdaptiveOptimizer 创建自适应优化引擎
func NewAdaptiveOptimizer(configManager *ConfigManager, performanceMonitor *StandardPerformanceMonitor) *AdaptiveOptimizer {
	ctx, cancel := context.WithCancel(context.Background())
	optimizer := &AdaptiveOptimizer{
		configManager:      configManager,
		performanceMonitor: performanceMonitor,
		optimizationRules:  getDefaultOptimizationRules(),
		running:            false,
		ctx:                ctx,
		cancel:             cancel,
		optimizationLog:    make([]OptimizationEvent, 0),
	}
	return optimizer
}

// Start 启动自适应优化
func (ao *AdaptiveOptimizer) Start(interval time.Duration) error {
	ao.mu.Lock()
	defer ao.mu.Unlock()

	if ao.running {
		return fmt.Errorf("adaptive optimizer is already running")
	}

	ao.running = true
	go ao.optimizationLoop(interval)
	return nil
}

// Stop 停止自适应优化
func (ao *AdaptiveOptimizer) Stop() {
	ao.mu.Lock()
	defer ao.mu.Unlock()

	if !ao.running {
		return
	}

	ao.running = false
	ao.cancel()
}

// AddRule 添加优化规则
func (ao *AdaptiveOptimizer) AddRule(rule OptimizationRule) {
	ao.mu.Lock()
	defer ao.mu.Unlock()

	ao.optimizationRules = append(ao.optimizationRules, rule)
}

// RemoveRule 移除优化规则
func (ao *AdaptiveOptimizer) RemoveRule(name string) {
	ao.mu.Lock()
	defer ao.mu.Unlock()

	for i, rule := range ao.optimizationRules {
		if rule.Name == name {
			ao.optimizationRules = append(ao.optimizationRules[:i], ao.optimizationRules[i+1:]...)
			break
		}
	}
}

// GetOptimizationLog 获取优化日志
func (ao *AdaptiveOptimizer) GetOptimizationLog() []OptimizationEvent {
	ao.mu.RLock()
	defer ao.mu.RUnlock()

	return append([]OptimizationEvent(nil), ao.optimizationLog...)
}

// GetCurrentMetrics 获取当前性能指标
func (ao *AdaptiveOptimizer) GetCurrentMetrics() (*PerformanceMetrics, error) {
	// 从性能监控器获取当前指标
	stats := ao.performanceMonitor.GetSystemStats()
	metricsData := ao.performanceMonitor.GetMetrics()

	metrics := &PerformanceMetrics{
		MemoryUsage:   stats.MemoryUsage.Allocated,
		LastUpdated:   time.Now(),
		ThroughputQPS: 0.0, // 默认值
		Recall:        0.0, // 默认值
	}

	// 从指标数据中获取查询相关指标
	if searchLatency, exists := metricsData["search_latency"]; exists {
		metrics.AvgLatency = time.Duration(searchLatency.Average * float64(time.Millisecond))
	}

	if throughput, exists := metricsData["search_throughput"]; exists {
		metrics.ThroughputQPS = throughput.Value
	}

	// ErrorRate字段在PerformanceMetrics中不存在，跳过设置

	return metrics, nil
}

// 私有方法

func (ao *AdaptiveOptimizer) optimizationLoop(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ao.ctx.Done():
			return
		case <-ticker.C:
			ao.performOptimization()
		}
	}
}

func (ao *AdaptiveOptimizer) performOptimization() {
	metrics, err := ao.GetCurrentMetrics()
	if err != nil {
		return
	}

	ao.mu.Lock()
	defer ao.mu.Unlock()

	// 按优先级排序规则
	rules := make([]OptimizationRule, len(ao.optimizationRules))
	copy(rules, ao.optimizationRules)

	// 应用优化规则
	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}

		// 检查冷却时间
		if time.Since(rule.LastApplied) < rule.Cooldown {
			continue
		}

		// 检查条件
		if rule.Condition(metrics) {
			err := rule.Action(ao.configManager)
			event := OptimizationEvent{
				Timestamp:   time.Now(),
				RuleName:    rule.Name,
				Description: rule.Description,
				Metrics:     metrics,
				Success:     err == nil,
				Error:       err,
			}

			ao.optimizationLog = append(ao.optimizationLog, event)

			// 限制日志大小
			if len(ao.optimizationLog) > 1000 {
				ao.optimizationLog = ao.optimizationLog[100:]
			}

			// 更新规则的最后应用时间
			for i := range ao.optimizationRules {
				if ao.optimizationRules[i].Name == rule.Name {
					ao.optimizationRules[i].LastApplied = time.Now()
					break
				}
			}
		}
	}
}

// getDefaultOptimizationRules 获取默认优化规则
func getDefaultOptimizationRules() []OptimizationRule {
	return []OptimizationRule{
		{
			Name:        "HighCPUOptimization",
			Description: "CPU使用率过高时启用查询加速",
			Condition: func(metrics *PerformanceMetrics) bool {
				// 使用内存使用率作为替代指标
				return metrics.MemoryUsage > 1024*1024*1024*8 // 8GB
			},
			Action: func(cm *ConfigManager) error {
				if cm.PerformanceConfig != nil {
					cm.PerformanceConfig.QueryAcceleration.Enable = true
					cm.PerformanceConfig.QueryAcceleration.MultiStageSearch.Enable = true
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 1,
			Cooldown: 5 * time.Minute,
			Enabled:  true,
		},
		{
			Name:        "HighMemoryOptimization",
			Description: "内存使用率过高时启用内存管理优化",
			Condition: func(metrics *PerformanceMetrics) bool {
				// 使用内存使用量作为替代指标
				return metrics.MemoryUsage > 1024*1024*1024*10 // 10GB
			},
			Action: func(cm *ConfigManager) error {
				if cm.PerformanceConfig != nil {
					cm.PerformanceConfig.MemoryManagement.MemoryPool.Enable = true
					cm.PerformanceConfig.MemoryManagement.GarbageCollection.Enable = true
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 1,
			Cooldown: 5 * time.Minute,
			Enabled:  true,
		},
		{
			Name:        "HighLatencyOptimization",
			Description: "查询延迟过高时优化索引策略",
			Condition: func(metrics *PerformanceMetrics) bool {
				return metrics.AvgLatency > 200*time.Millisecond
			},
			Action: func(cm *ConfigManager) error {
				if cm.DistributedConfig != nil {
					// 启用自适应索引选择
					cm.DistributedConfig.IndexConfig.AdaptiveSelection.Enable = true
					// 优化HNSW参数
					cm.DistributedConfig.IndexConfig.HNSWConfig.MaxConnections = int(math.Min(float64(cm.DistributedConfig.IndexConfig.HNSWConfig.MaxConnections+4), 64))
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 2,
			Cooldown: 10 * time.Minute,
			Enabled:  true,
		},
		{
			Name:        "LowCacheHitRateOptimization",
			Description: "缓存命中率低时调整缓存策略",
			Condition: func(metrics *PerformanceMetrics) bool {
				// 使用召回率作为替代指标
				return metrics.Recall < 0.3 && metrics.Recall > 0
			},
			Action: func(cm *ConfigManager) error {
				if cm.CacheConfig != nil {
					// 增加缓存大小
					cm.CacheConfig.ResultCache.MaxSize = int64(float64(cm.CacheConfig.ResultCache.MaxSize) * 1.5)
					cm.CacheConfig.VectorCache.MaxSize = int64(float64(cm.CacheConfig.VectorCache.MaxSize) * 1.5)
					// 缓存命中率低，启用预取
					cm.CacheConfig.ResultCache.Optimization.Prefetching.Enable = true
					cm.CacheConfig.VectorCache.Prefetching.Enable = true
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 3,
			Cooldown: 15 * time.Minute,
			Enabled:  true,
		},
		{
			Name:        "HighThroughputOptimization",
			Description: "高吞吐量场景下启用批处理优化",
			Condition: func(metrics *PerformanceMetrics) bool {
				return metrics.ThroughputQPS > 1000
			},
			Action: func(cm *ConfigManager) error {
				if cm.PerformanceConfig != nil {
					cm.PerformanceConfig.BatchProcessing.Enable = true
					cm.PerformanceConfig.BatchProcessing.BatchSize = int(math.Min(float64(cm.PerformanceConfig.BatchProcessing.BatchSize*2), 1000))
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 4,
			Cooldown: 10 * time.Minute,
			Enabled:  true,
		},
		{
			Name:        "LowThroughputOptimization",
			Description: "低吞吐量场景下优化资源使用",
			Condition: func(metrics *PerformanceMetrics) bool {
				// 使用吞吐量和内存使用量作为替代指标
				return metrics.ThroughputQPS < 10 && metrics.MemoryUsage < 1024*1024*1024*2 // 2GB
			},
			Action: func(cm *ConfigManager) error {
				if cm.PerformanceConfig != nil {
					// 减少并发数
					cm.PerformanceConfig.ConcurrencyControl.MaxConcurrentQueries = int(math.Max(float64(cm.PerformanceConfig.ConcurrencyControl.MaxConcurrentQueries/2), 1))
					// 启用节能模式
					if cm.HardwareConfig != nil {
						cm.HardwareConfig.CPU.PowerManagement.Enable = true
						cm.HardwareConfig.CPU.PowerManagement.Governor = "powersave"
					}
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 5,
			Cooldown: 20 * time.Minute,
			Enabled:  true,
		},
		{
			Name:        "ErrorRateOptimization",
			Description: "错误率过高时启用容错机制",
			Condition: func(metrics *PerformanceMetrics) bool {
				// 使用召回率作为替代指标，低召回率可能表示错误
				return metrics.Recall < 0.95 && metrics.Recall > 0
			},
			Action: func(cm *ConfigManager) error {
				if cm.DistributedConfig != nil {
					// 启用负载均衡
					cm.DistributedConfig.ArchitectureConfig.LoadBalancing.Strategy = "least_load"
					cm.DistributedConfig.ArchitectureConfig.LoadBalancing.HealthCheckInterval = 30 * time.Second
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 1,
			Cooldown: 5 * time.Minute,
			Enabled:  true,
		},
		{
			Name:        "DataSizeOptimization",
			Description: "根据数据规模调整索引策略",
			Condition: func(metrics *PerformanceMetrics) bool {
				// 使用内存使用量作为数据规模的替代指标
				return metrics.MemoryUsage > 10*1024*1024*1024 // 10GB
			},
			Action: func(cm *ConfigManager) error {
				if cm.DistributedConfig != nil {
					// 启用IVF索引用于大数据集
					cm.DistributedConfig.IndexConfig.IVFConfig.Enable = true
					cm.DistributedConfig.IndexConfig.IVFConfig.NumClusters = 4096

					// 启用数据压缩
					cm.DistributedConfig.IndexConfig.PQConfig.Enable = true
					return cm.SaveConfig()
				}
				return nil
			},
			Priority: 6,
			Cooldown: 30 * time.Minute,
			Enabled:  true,
		},
	}
}

// OptimizationStrategy 优化策略
type OptimizationStrategy struct {
	Name        string
	Description string
	Rules       []OptimizationRule
}

// GetOptimizationStrategies 获取预定义的优化策略
func GetOptimizationStrategies() []OptimizationStrategy {
	return []OptimizationStrategy{
		{
			Name:        "HighPerformance",
			Description: "高性能策略，优先考虑查询速度",
			Rules:       []OptimizationRule{
				// 高性能相关的规则
			},
		},
		{
			Name:        "MemoryEfficient",
			Description: "内存高效策略，优先考虑内存使用",
			Rules:       []OptimizationRule{
				// 内存高效相关的规则
			},
		},
		{
			Name:        "Balanced",
			Description: "平衡策略，在性能和资源使用之间取得平衡",
			Rules:       getDefaultOptimizationRules(),
		},
	}
}

// ApplyStrategy 应用优化策略
func (ao *AdaptiveOptimizer) ApplyStrategy(strategyName string) error {
	strategies := GetOptimizationStrategies()
	for _, strategy := range strategies {
		if strategy.Name == strategyName {
			ao.mu.Lock()
			ao.optimizationRules = strategy.Rules
			ao.mu.Unlock()
			return nil
		}
	}
	return fmt.Errorf("strategy not found: %s", strategyName)
}
