package optimization

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"VectorSphere/src/vector"
	"context"
	"sync"
	"time"
)

// Connector 高吞吐量连接器
// 用于将高吞吐量优化器与向量数据库集成
type Connector struct {
	vectorDB                *vector.VectorDB
	highThroughputOptimizer *HighThroughputOptimizer
	indexOptimizer          *IndexOptimizer
	cacheOptimizer          *CacheOptimizer
	performanceMonitor      *PerformanceMonitor
	config                  *ConnectorConfig
	mu                      sync.RWMutex
	isInitialized           bool
}

// ConnectorConfig 连接器配置
type ConnectorConfig struct {
	// 最大并发搜索数
	MaxConcurrentSearches int `json:"max_concurrent_searches"`

	// 批处理大小
	BatchSize int `json:"batch_size"`

	// 是否启用GPU加速
	EnableGPU bool `json:"enable_gpu"`

	// 是否启用缓存
	EnableCache bool `json:"enable_cache"`

	// 是否启用索引优化
	EnableIndexOptimization bool `json:"enable_index_optimization"`

	// 是否启用性能监控
	EnablePerformanceMonitoring bool `json:"enable_performance_monitoring"`

	// 默认搜索超时时间
	DefaultSearchTimeout time.Duration `json:"default_search_timeout"`

	// 默认搜索质量级别 (0-1.0)
	DefaultQualityLevel float64 `json:"default_quality_level"`
}

// SearchOptions 搜索选项
type SearchOptions struct {
	// 搜索超时时间
	Timeout time.Duration

	// 质量级别 (0-1.0)
	QualityLevel float64

	// 是否启用GPU
	EnableGPU bool

	// 是否启用缓存
	EnableCache bool

	// 强制使用特定策略
	ForceStrategy string

	// IVF探测数
	Nprobe int

	// 批处理大小
	BatchSize int
}

// NewConnector 创建连接器
func NewConnector(vectorDB *vector.VectorDB, config *ConnectorConfig) *Connector {
	if config == nil {
		config = getDefaultConnectorConfig()
	}

	return &Connector{
		vectorDB:      vectorDB,
		config:        config,
		isInitialized: false,
	}
}

// getDefaultConnectorConfig 获取默认连接器配置
func getDefaultConnectorConfig() *ConnectorConfig {
	return &ConnectorConfig{
		MaxConcurrentSearches:       100,
		BatchSize:                   64,
		EnableGPU:                   true,
		EnableCache:                 true,
		EnableIndexOptimization:     true,
		EnablePerformanceMonitoring: true,
		DefaultSearchTimeout:        time.Second * 2,
		DefaultQualityLevel:         0.9,
	}
}

// Initialize 初始化连接器
func (c *Connector) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.isInitialized {
		return nil
	}

	// 创建高吞吐量优化器
	c.highThroughputOptimizer = NewHighThroughputOptimizer(c.vectorDB, &HighThroughputConfig{
		MaxConcurrentSearches: c.config.MaxConcurrentSearches,
		BatchSize:             c.config.BatchSize,
		EnableGPU:             c.config.EnableGPU,
		EnableCache:           c.config.EnableCache,
	})

	// 创建索引优化器
	if c.config.EnableIndexOptimization {
		c.indexOptimizer = NewIndexOptimizer(c.vectorDB, nil)
		c.indexOptimizer.StartOptimizationScheduler()
	}

	// 创建缓存优化器
	if c.config.EnableCache {
		c.cacheOptimizer = NewCacheOptimizer(c.vectorDB)
	}

	// 创建性能监控器
	if c.config.EnablePerformanceMonitoring {
		c.performanceMonitor = NewPerformanceMonitor(nil)
		c.performanceMonitor.Start()
	}

	c.isInitialized = true
	log.Info("高吞吐量连接器初始化完成")
	return nil
}

// Search 单向量搜索
// Search 向量搜索
func (c *Connector) Search(ctx context.Context, vector []float64, k int, options *SearchOptions) ([]entity.Result, error) {
	if !c.isInitialized {
		if err := c.Initialize(); err != nil {
			return nil, err
		}
	}

	// 设置默认选项
	if options == nil {
		options = &SearchOptions{
			Timeout:      c.config.DefaultSearchTimeout,
			QualityLevel: c.config.DefaultQualityLevel,
			EnableGPU:    c.config.EnableGPU,
			EnableCache:  c.config.EnableCache,
		}
	}

	// 记录开始时间
	startTime := time.Now()

	// 执行优化搜索
	results, err := c.highThroughputOptimizer.OptimizedSearch(ctx, vector, k, options)

	// 记录性能数据
	if c.performanceMonitor != nil {
		c.performanceMonitor.RecordSearchPerformance(1, startTime, time.Since(startTime), err == nil)
	}

	return results, err
}

// BatchSearch 批量向量搜索
func (c *Connector) BatchSearch(ctx context.Context, vectors [][]float64, k int, options *SearchOptions) ([][]entity.Result, error) {
	if !c.isInitialized {
		if err := c.Initialize(); err != nil {
			return nil, err
		}
	}

	// 设置默认选项
	if options == nil {
		options = &SearchOptions{
			Timeout:      c.config.DefaultSearchTimeout,
			QualityLevel: c.config.DefaultQualityLevel,
			EnableGPU:    c.config.EnableGPU,
			EnableCache:  c.config.EnableCache,
			BatchSize:    c.config.BatchSize,
		}
	}

	startTime := time.Now()

	// 执行批量搜索
	results, err := c.highThroughputOptimizer.BatchSearch(ctx, vectors, k, options)

	// 记录性能数据
	if c.performanceMonitor != nil {
		c.performanceMonitor.RecordBatchSearchPerformance(len(vectors), startTime, time.Since(startTime), err == nil)
	}

	return results, err
}

// OptimizeCache 优化缓存
func (c *Connector) OptimizeCache() error {
	if !c.isInitialized {
		if err := c.Initialize(); err != nil {
			return err
		}
	}

	if c.cacheOptimizer == nil {
		return nil
	}

	return c.cacheOptimizer.Optimize()
}

// OptimizeIndex 优化索引
func (c *Connector) OptimizeIndex() error {
	if !c.isInitialized {
		if err := c.Initialize(); err != nil {
			return err
		}
	}

	if c.indexOptimizer == nil {
		return nil
	}

	c.indexOptimizer.CheckAndOptimize()
	return nil
}

// GetPerformanceMetrics 获取性能指标
func (c *Connector) GetPerformanceMetrics() (*PerformanceMetrics, error) {
	if !c.isInitialized {
		if err := c.Initialize(); err != nil {
			return nil, err
		}
	}

	if c.performanceMonitor == nil {
		return nil, nil
	}

	return c.performanceMonitor.GetMetrics(), nil
}

// GetIndexStats 获取索引统计
func (c *Connector) GetIndexStats() (*IndexStats, error) {
	if !c.isInitialized {
		if err := c.Initialize(); err != nil {
			return nil, err
		}
	}

	if c.indexOptimizer == nil {
		return nil, nil
	}

	return c.indexOptimizer.GetIndexStats(), nil
}

// RecordDataChange 记录数据变化
func (c *Connector) RecordDataChange(count int) {
	if !c.isInitialized || c.indexOptimizer == nil {
		return
	}

	c.indexOptimizer.RecordDataChange(count)
}

// Close 关闭连接器
func (c *Connector) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.isInitialized {
		return nil
	}

	// 关闭性能监控器
	if c.performanceMonitor != nil {
		c.performanceMonitor.StopMonitoring()
	}

	c.isInitialized = false
	log.Info("高吞吐量连接器已关闭")
	return nil
}
