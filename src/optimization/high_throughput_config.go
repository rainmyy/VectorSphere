package optimization

import (
	"time"
)

// HighThroughputConfig 高吞吐量配置
type HighThroughputConfig struct {
	// 最大并发搜索数
	MaxConcurrentSearches int `json:"max_concurrent_searches"`
	
	// 批处理大小
	BatchSize int `json:"batch_size"`
	
	// 是否启用GPU加速
	EnableGPU bool `json:"enable_gpu"`
	
	// 是否启用缓存
	EnableCache bool `json:"enable_cache"`
	
	// 工作池大小
	WorkerPoolSize int `json:"worker_pool_size"`
	
	// 任务队列容量
	TaskQueueCapacity int `json:"task_queue_capacity"`
	
	// 最大批处理等待时间
	MaxBatchWaitTime time.Duration `json:"max_batch_wait_time"`
	
	// 自适应批处理大小
	AdaptiveBatchSize bool `json:"adaptive_batch_size"`
	
	// 最小批处理大小
	MinBatchSize int `json:"min_batch_size"`
	
	// 最大批处理大小
	MaxBatchSize int `json:"max_batch_size"`
	
	// 批处理大小调整因子
	BatchSizeAdjustFactor float64 `json:"batch_size_adjust_factor"`
	
	// 性能采样间隔
	PerformanceSamplingInterval time.Duration `json:"performance_sampling_interval"`
	
	// 自适应优化间隔
	AdaptiveOptimizationInterval time.Duration `json:"adaptive_optimization_interval"`
	
	// 是否启用优先级队列
	EnablePriorityQueue bool `json:"enable_priority_queue"`
	
	// 是否启用请求合并
	EnableRequestMerging bool `json:"enable_request_merging"`
	
	// 请求合并窗口
	RequestMergeWindow time.Duration `json:"request_merge_window"`
	
	// 是否启用结果缓存
	EnableResultCache bool `json:"enable_result_cache"`
	
	// 结果缓存过期时间
	ResultCacheTTL time.Duration `json:"result_cache_ttl"`
	
	// 结果缓存最大条目数
	ResultCacheMaxEntries int `json:"result_cache_max_entries"`
	
	// 是否启用自适应超时
	EnableAdaptiveTimeout bool `json:"enable_adaptive_timeout"`
	
	// 基础超时时间
	BaseTimeout time.Duration `json:"base_timeout"`
	
	// 最大超时时间
	MaxTimeout time.Duration `json:"max_timeout"`
	
	// 超时调整因子
	TimeoutAdjustFactor float64 `json:"timeout_adjust_factor"`
	
	// 是否启用负载均衡
	EnableLoadBalancing bool `json:"enable_load_balancing"`
	
	// 负载均衡策略
	LoadBalancingStrategy string `json:"load_balancing_strategy"`
	
	// 是否启用熔断器
	EnableCircuitBreaker bool `json:"enable_circuit_breaker"`
	
	// 熔断阈值
	CircuitBreakerThreshold float64 `json:"circuit_breaker_threshold"`
	
	// 熔断恢复时间
	CircuitBreakerRecoveryTime time.Duration `json:"circuit_breaker_recovery_time"`
	
	// 是否启用自适应索引选择
	EnableAdaptiveIndexSelection bool `json:"enable_adaptive_index_selection"`
	
	// 是否启用自适应距离函数
	EnableAdaptiveDistanceFunction bool `json:"enable_adaptive_distance_function"`
	
	// 是否启用自适应精排
	EnableAdaptiveFineRanking bool `json:"enable_adaptive_fine_ranking"`
	
	// 是否启用向量压缩
	EnableVectorCompression bool `json:"enable_vector_compression"`
	
	// 向量压缩方法
	VectorCompressionMethod string `json:"vector_compression_method"`
	
	// 是否启用预取
	EnablePrefetching bool `json:"enable_prefetching"`
	
	// 预取因子
	PrefetchFactor float64 `json:"prefetch_factor"`
	
	// 是否启用查询重写
	EnableQueryRewriting bool `json:"enable_query_rewriting"`
	
	// 是否启用查询优化
	EnableQueryOptimization bool `json:"enable_query_optimization"`
	
	// 是否启用查询计划缓存
	EnableQueryPlanCache bool `json:"enable_query_plan_cache"`
	
	// 查询计划缓存大小
	QueryPlanCacheSize int `json:"query_plan_cache_size"`
	
	// 是否启用硬件感知优化
	EnableHardwareAwareOptimization bool `json:"enable_hardware_aware_optimization"`
}

// GetDefaultHighThroughputConfig 获取默认高吞吐量配置
func GetDefaultHighThroughputConfig() *HighThroughputConfig {
	return &HighThroughputConfig{
		MaxConcurrentSearches:          100,
		BatchSize:                      64,
		EnableGPU:                      true,
		EnableCache:                    true,
		WorkerPoolSize:                 8,
		TaskQueueCapacity:              1000,
		MaxBatchWaitTime:               time.Millisecond * 50,
		AdaptiveBatchSize:              true,
		MinBatchSize:                   16,
		MaxBatchSize:                   256,
		BatchSizeAdjustFactor:          1.5,
		PerformanceSamplingInterval:    time.Second * 10,
		AdaptiveOptimizationInterval:   time.Minute * 5,
		EnablePriorityQueue:            true,
		EnableRequestMerging:           true,
		RequestMergeWindow:             time.Millisecond * 20,
		EnableResultCache:              true,
		ResultCacheTTL:                 time.Minute * 10,
		ResultCacheMaxEntries:          10000,
		EnableAdaptiveTimeout:          true,
		BaseTimeout:                    time.Second * 1,
		MaxTimeout:                     time.Second * 10,
		TimeoutAdjustFactor:            1.5,
		EnableLoadBalancing:            true,
		LoadBalancingStrategy:          "least_loaded",
		EnableCircuitBreaker:           true,
		CircuitBreakerThreshold:        0.5,
		CircuitBreakerRecoveryTime:     time.Second * 30,
		EnableAdaptiveIndexSelection:   true,
		EnableAdaptiveDistanceFunction: true,
		EnableAdaptiveFineRanking:      true,
		EnableVectorCompression:         false,
		VectorCompressionMethod:         "pq",
		EnablePrefetching:              false,
		PrefetchFactor:                 1.5,
		EnableQueryRewriting:           false,
		EnableQueryOptimization:        true,
		EnableQueryPlanCache:           true,
		QueryPlanCacheSize:             1000,
		EnableHardwareAwareOptimization: true,
	}
}

// MergeWithDefault 与默认配置合并
func (c *HighThroughputConfig) MergeWithDefault() *HighThroughputConfig {
	defaultConfig := GetDefaultHighThroughputConfig()
	
	// 如果配置项为零值，则使用默认值
	if c.MaxConcurrentSearches <= 0 {
		c.MaxConcurrentSearches = defaultConfig.MaxConcurrentSearches
	}
	
	if c.BatchSize <= 0 {
		c.BatchSize = defaultConfig.BatchSize
	}
	
	if c.WorkerPoolSize <= 0 {
		c.WorkerPoolSize = defaultConfig.WorkerPoolSize
	}
	
	if c.TaskQueueCapacity <= 0 {
		c.TaskQueueCapacity = defaultConfig.TaskQueueCapacity
	}
	
	if c.MaxBatchWaitTime <= 0 {
		c.MaxBatchWaitTime = defaultConfig.MaxBatchWaitTime
	}
	
	if c.MinBatchSize <= 0 {
		c.MinBatchSize = defaultConfig.MinBatchSize
	}
	
	if c.MaxBatchSize <= 0 {
		c.MaxBatchSize = defaultConfig.MaxBatchSize
	}
	
	if c.BatchSizeAdjustFactor <= 0 {
		c.BatchSizeAdjustFactor = defaultConfig.BatchSizeAdjustFactor
	}
	
	if c.PerformanceSamplingInterval <= 0 {
		c.PerformanceSamplingInterval = defaultConfig.PerformanceSamplingInterval
	}
	
	if c.AdaptiveOptimizationInterval <= 0 {
		c.AdaptiveOptimizationInterval = defaultConfig.AdaptiveOptimizationInterval
	}
	
	if c.RequestMergeWindow <= 0 {
		c.RequestMergeWindow = defaultConfig.RequestMergeWindow
	}
	
	if c.ResultCacheTTL <= 0 {
		c.ResultCacheTTL = defaultConfig.ResultCacheTTL
	}
	
	if c.ResultCacheMaxEntries <= 0 {
		c.ResultCacheMaxEntries = defaultConfig.ResultCacheMaxEntries
	}
	
	if c.BaseTimeout <= 0 {
		c.BaseTimeout = defaultConfig.BaseTimeout
	}
	
	if c.MaxTimeout <= 0 {
		c.MaxTimeout = defaultConfig.MaxTimeout
	}
	
	if c.TimeoutAdjustFactor <= 0 {
		c.TimeoutAdjustFactor = defaultConfig.TimeoutAdjustFactor
	}
	
	if c.LoadBalancingStrategy == "" {
		c.LoadBalancingStrategy = defaultConfig.LoadBalancingStrategy
	}
	
	if c.CircuitBreakerThreshold <= 0 {
		c.CircuitBreakerThreshold = defaultConfig.CircuitBreakerThreshold
	}
	
	if c.CircuitBreakerRecoveryTime <= 0 {
		c.CircuitBreakerRecoveryTime = defaultConfig.CircuitBreakerRecoveryTime
	}
	
	if c.VectorCompressionMethod == "" {
		c.VectorCompressionMethod = defaultConfig.VectorCompressionMethod
	}
	
	if c.PrefetchFactor <= 0 {
		c.PrefetchFactor = defaultConfig.PrefetchFactor
	}
	
	if c.QueryPlanCacheSize <= 0 {
		c.QueryPlanCacheSize = defaultConfig.QueryPlanCacheSize
	}
	
	return c
}

// Validate 验证配置
func (c *HighThroughputConfig) Validate() error {
	// 验证批处理大小范围
	if c.MinBatchSize > c.MaxBatchSize {
		c.MinBatchSize = c.MaxBatchSize / 2
	}
	
	// 验证批处理大小
	if c.BatchSize < c.MinBatchSize {
		c.BatchSize = c.MinBatchSize
	} else if c.BatchSize > c.MaxBatchSize {
		c.BatchSize = c.MaxBatchSize
	}
	
	// 验证工作池大小
	if c.WorkerPoolSize <= 0 {
		c.WorkerPoolSize = 1
	} else if c.WorkerPoolSize > 64 {
		c.WorkerPoolSize = 64
	}
	
	// 验证任务队列容量
	if c.TaskQueueCapacity <= 0 {
		c.TaskQueueCapacity = 100
	}
	
	// 验证超时时间
	if c.BaseTimeout > c.MaxTimeout {
		c.BaseTimeout = c.MaxTimeout / 2
	}
	
	return nil
}

// OptimizeForHardware 根据硬件优化配置
func (c *HighThroughputConfig) OptimizeForHardware(cpuCores int, memoryGB int, hasGPU bool, gpuMemoryGB int) *HighThroughputConfig {
	// 根据CPU核心数优化工作池大小
	c.WorkerPoolSize = cpuCores - 1
	if c.WorkerPoolSize < 1 {
		c.WorkerPoolSize = 1
	} else if c.WorkerPoolSize > 32 {
		c.WorkerPoolSize = 32
	}
	
	// 根据内存大小优化缓存大小
	c.ResultCacheMaxEntries = memoryGB * 1000
	if c.ResultCacheMaxEntries < 1000 {
		c.ResultCacheMaxEntries = 1000
	} else if c.ResultCacheMaxEntries > 100000 {
		c.ResultCacheMaxEntries = 100000
	}
	
	// 根据GPU可用性优化GPU相关配置
	c.EnableGPU = hasGPU
	
	if hasGPU {
		// 根据GPU内存大小优化批处理大小
		c.MaxBatchSize = gpuMemoryGB * 32
		if c.MaxBatchSize < 64 {
			c.MaxBatchSize = 64
		} else if c.MaxBatchSize > 1024 {
			c.MaxBatchSize = 1024
		}
		
		c.BatchSize = c.MaxBatchSize / 2
		c.MinBatchSize = c.MaxBatchSize / 4
	} else {
		// CPU优化的批处理大小
		c.MaxBatchSize = 128
		c.BatchSize = 64
		c.MinBatchSize = 32
	}
	
	return c
}

// OptimizeForWorkload 根据工作负载优化配置
func (c *HighThroughputConfig) OptimizeForWorkload(avgQueriesPerSecond float64, avgVectorDimension int, avgResultsPerQuery int) *HighThroughputConfig {
	// 根据查询率优化并发数
	c.MaxConcurrentSearches = int(avgQueriesPerSecond * 2)
	if c.MaxConcurrentSearches < 10 {
		c.MaxConcurrentSearches = 10
	} else if c.MaxConcurrentSearches > 1000 {
		c.MaxConcurrentSearches = 1000
	}
	
	// 根据向量维度优化批处理大小
	if avgVectorDimension > 1000 {
		// 高维向量，减小批处理大小
		c.BatchSize = c.BatchSize / 2
		if c.BatchSize < c.MinBatchSize {
			c.BatchSize = c.MinBatchSize
		}
	} else if avgVectorDimension < 100 {
		// 低维向量，增大批处理大小
		c.BatchSize = c.BatchSize * 2
		if c.BatchSize > c.MaxBatchSize {
			c.BatchSize = c.MaxBatchSize
		}
	}
	
	// 根据结果数量优化缓存策略
	if avgResultsPerQuery > 100 {
		// 大结果集，减小缓存条目数量
		c.ResultCacheMaxEntries = c.ResultCacheMaxEntries / 2
	}
	
	// 根据查询率优化合并窗口
	if avgQueriesPerSecond > 100 {
		// 高查询率，减小合并窗口
		c.RequestMergeWindow = time.Millisecond * 10
	} else if avgQueriesPerSecond < 10 {
		// 低查询率，增大合并窗口
		c.RequestMergeWindow = time.Millisecond * 50
	}
	
	return c
}

// ToMap 转换为Map
func (c *HighThroughputConfig) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"max_concurrent_searches":           c.MaxConcurrentSearches,
		"batch_size":                         c.BatchSize,
		"enable_gpu":                         c.EnableGPU,
		"enable_cache":                       c.EnableCache,
		"worker_pool_size":                   c.WorkerPoolSize,
		"task_queue_capacity":                c.TaskQueueCapacity,
		"max_batch_wait_time_ms":             c.MaxBatchWaitTime.Milliseconds(),
		"adaptive_batch_size":                c.AdaptiveBatchSize,
		"min_batch_size":                     c.MinBatchSize,
		"max_batch_size":                     c.MaxBatchSize,
		"batch_size_adjust_factor":           c.BatchSizeAdjustFactor,
		"performance_sampling_interval_sec":  c.PerformanceSamplingInterval.Seconds(),
		"adaptive_optimization_interval_sec": c.AdaptiveOptimizationInterval.Seconds(),
		"enable_priority_queue":              c.EnablePriorityQueue,
		"enable_request_merging":             c.EnableRequestMerging,
		"request_merge_window_ms":            c.RequestMergeWindow.Milliseconds(),
		"enable_result_cache":                c.EnableResultCache,
		"result_cache_ttl_sec":               c.ResultCacheTTL.Seconds(),
		"result_cache_max_entries":           c.ResultCacheMaxEntries,
		"enable_adaptive_timeout":            c.EnableAdaptiveTimeout,
		"base_timeout_ms":                    c.BaseTimeout.Milliseconds(),
		"max_timeout_ms":                     c.MaxTimeout.Milliseconds(),
		"timeout_adjust_factor":              c.TimeoutAdjustFactor,
		"enable_load_balancing":              c.EnableLoadBalancing,
		"load_balancing_strategy":            c.LoadBalancingStrategy,
		"enable_circuit_breaker":             c.EnableCircuitBreaker,
		"circuit_breaker_threshold":          c.CircuitBreakerThreshold,
		"circuit_breaker_recovery_time_sec":  c.CircuitBreakerRecoveryTime.Seconds(),
		"enable_adaptive_index_selection":    c.EnableAdaptiveIndexSelection,
		"enable_adaptive_distance_function":  c.EnableAdaptiveDistanceFunction,
		"enable_adaptive_fine_ranking":       c.EnableAdaptiveFineRanking,
		"enable_vector_compression":          c.EnableVectorCompression,
		"vector_compression_method":          c.VectorCompressionMethod,
		"enable_prefetching":                 c.EnablePrefetching,
		"prefetch_factor":                    c.PrefetchFactor,
		"enable_query_rewriting":             c.EnableQueryRewriting,
		"enable_query_optimization":          c.EnableQueryOptimization,
		"enable_query_plan_cache":            c.EnableQueryPlanCache,
		"query_plan_cache_size":              c.QueryPlanCacheSize,
		"enable_hardware_aware_optimization": c.EnableHardwareAwareOptimization,
	}
}