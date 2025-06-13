package optimization

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"VectorSphere/src/vector"
	"context"
	"runtime"
	"sync"
	"time"
)

// IndexOptimizer 索引优化器
type IndexOptimizer struct {
	vectorDB           *vector.VectorDB
	config             *IndexOptimizerConfig
	lastOptimization   time.Time
	indexStats         *IndexStats
	mu                 sync.RWMutex
	optimizationActive bool
}

// IndexOptimizerConfig 索引优化器配置
type IndexOptimizerConfig struct {
	// 自动重建索引的阈值（数据变化比例）
	ReindexThreshold float64 `json:"reindex_threshold"`

	// 自动优化间隔
	OptimizationInterval time.Duration `json:"optimization_interval"`

	// 是否启用自动重建索引
	AutoReindex bool `json:"auto_reindex"`

	// 是否启用索引优化
	IndexOptimization bool `json:"index_optimization"`

	// 最大索引构建时间
	MaxIndexBuildTime time.Duration `json:"max_index_build_time"`

	// 索引质量阈值
	IndexQualityThreshold float64 `json:"index_quality_threshold"`

	// 是否启用增量索引
	EnableIncrementalIndex bool `json:"enable_incremental_index"`

	// 增量索引阈值（新增数据比例）
	IncrementalThreshold float64 `json:"incremental_threshold"`
}

// IndexStats 索引统计
type IndexStats struct {
	// 索引大小（字节）
	IndexSize int64 `json:"index_size"`

	// 索引构建时间
	BuildTime time.Duration `json:"build_time"`

	// 索引质量（召回率）
	Quality float64 `json:"quality"`

	// 索引类型
	IndexType string `json:"index_type"`

	// 上次重建时间
	LastRebuildTime time.Time `json:"last_rebuild_time"`

	// 数据变化计数（自上次重建）
	DataChanges int64 `json:"data_changes"`

	// 总数据量
	TotalVectors int `json:"total_vectors"`

	// 索引参数
	Parameters map[string]interface{} `json:"parameters"`
}

// NewIndexOptimizer 创建索引优化器
func NewIndexOptimizer(vectorDB *vector.VectorDB, config *IndexOptimizerConfig) *IndexOptimizer {
	if config == nil {
		config = getDefaultIndexOptimizerConfig()
	}

	return &IndexOptimizer{
		vectorDB:         vectorDB,
		config:           config,
		lastOptimization: time.Now(),
		indexStats:       &IndexStats{LastRebuildTime: time.Now()},
	}
}

// getDefaultIndexOptimizerConfig 获取默认索引优化器配置
func getDefaultIndexOptimizerConfig() *IndexOptimizerConfig {
	return &IndexOptimizerConfig{
		ReindexThreshold:       0.1, // 10%数据变化触发重建
		OptimizationInterval:   1 * time.Hour,
		AutoReindex:            true,
		IndexOptimization:      true,
		MaxIndexBuildTime:      30 * time.Minute,
		IndexQualityThreshold:  0.9, // 90%质量阈值
		EnableIncrementalIndex: true,
		IncrementalThreshold:   0.05, // 5%新增数据触发增量索引
	}
}

// StartOptimizationScheduler 启动优化调度器
func (io *IndexOptimizer) StartOptimizationScheduler() {
	go func() {
		ticker := time.NewTicker(10 * time.Minute) // 每10分钟检查一次
		defer ticker.Stop()

		for range ticker.C {
			io.CheckAndOptimize()
		}
	}()

	log.Info("索引优化调度器已启动")
}

// CheckAndOptimize 检查并优化索引
func (io *IndexOptimizer) CheckAndOptimize() {
	io.mu.Lock()
	if io.optimizationActive {
		io.mu.Unlock()
		return
	}
	io.optimizationActive = true
	io.mu.Unlock()

	defer func() {
		io.mu.Lock()
		io.optimizationActive = false
		io.mu.Unlock()
	}()

	// 检查是否需要优化
	if !io.shouldOptimize() {
		return
	}

	log.Info("开始索引优化...")

	// 更新索引统计
	io.updateIndexStats()

	// 检查是否需要重建索引
	if io.shouldRebuildIndex() {
		io.rebuildIndex()
		return
	}

	// 检查是否需要增量索引
	if io.shouldIncrementalIndex() {
		io.incrementalIndex()
		return
	}

	// 执行索引优化
	if io.config.IndexOptimization {
		io.optimizeIndex()
	}

	// 更新最后优化时间
	io.mu.Lock()
	io.lastOptimization = time.Now()
	io.mu.Unlock()

	log.Info("索引优化完成")
}

// shouldOptimize 是否应该优化
func (io *IndexOptimizer) shouldOptimize() bool {
	io.mu.RLock()
	defer io.mu.RUnlock()

	// 检查距离上次优化的时间
	if time.Since(io.lastOptimization) < io.config.OptimizationInterval {
		return false
	}

	return true
}

// updateIndexStats 更新索引统计
func (io *IndexOptimizer) updateIndexStats() {
	io.mu.Lock()
	defer io.mu.Unlock()

	// 获取向量数据库的统计信息
	dbStats := io.vectorDB.GetStats()

	// 更新索引统计
	io.indexStats.TotalVectors = dbStats.VectorCount
	io.indexStats.IndexSize = dbStats.IndexSize
	io.indexStats.Quality = dbStats.IndexQuality

	// 获取当前索引类型
	indexType := "unknown"
	if dbStats.UsingIVF {
		indexType = "IVF"
	} else if dbStats.UsingHNSW {
		indexType = "HNSW"
	} else if dbStats.UsingLSH {
		indexType = "LSH"
	} else if dbStats.UsingPQ {
		indexType = "PQ"
	}
	io.indexStats.IndexType = indexType

	// 获取索引参数
	io.indexStats.Parameters = make(map[string]interface{})
	if dbStats.UsingIVF {
		io.indexStats.Parameters["nlist"] = dbStats.IVFNlist
		io.indexStats.Parameters["nprobe"] = dbStats.IVFNprobe
	} else if dbStats.UsingHNSW {
		io.indexStats.Parameters["ef_construction"] = dbStats.HNSWEfConstruction
		io.indexStats.Parameters["m"] = dbStats.HNSWM
	}

	log.Info("索引统计: 类型=%s, 大小=%dMB, 质量=%.2f, 向量数=%d",
		io.indexStats.IndexType,
		io.indexStats.IndexSize/1024/1024,
		io.indexStats.Quality,
		io.indexStats.TotalVectors)
}

// shouldRebuildIndex 是否应该重建索引
func (io *IndexOptimizer) shouldRebuildIndex() bool {
	io.mu.RLock()
	defer io.mu.RUnlock()

	// 如果未启用自动重建，返回false
	if !io.config.AutoReindex {
		return false
	}

	// 检查数据变化比例
	changeRatio := float64(io.indexStats.DataChanges) / float64(io.indexStats.TotalVectors)
	if changeRatio > io.config.ReindexThreshold {
		log.Info("数据变化比例(%.2f%%)超过阈值(%.2f%%)，需要重建索引",
			changeRatio*100, io.config.ReindexThreshold*100)
		return true
	}

	// 检查索引质量
	if io.indexStats.Quality < io.config.IndexQualityThreshold {
		log.Info("索引质量(%.2f)低于阈值(%.2f)，需要重建索引",
			io.indexStats.Quality, io.config.IndexQualityThreshold)
		return true
	}

	return false
}

// shouldIncrementalIndex 是否应该增量索引
func (io *IndexOptimizer) shouldIncrementalIndex() bool {
	io.mu.RLock()
	defer io.mu.RUnlock()

	// 如果未启用增量索引，返回false
	if !io.config.EnableIncrementalIndex {
		return false
	}

	// 检查新增数据比例
	changeRatio := float64(io.indexStats.DataChanges) / float64(io.indexStats.TotalVectors)
	if changeRatio > io.config.IncrementalThreshold && changeRatio <= io.config.ReindexThreshold {
		log.Info("数据变化比例(%.2f%%)适合增量索引", changeRatio*100)
		return true
	}

	return false
}

// rebuildIndex 重建索引
func (io *IndexOptimizer) rebuildIndex() {
	log.Info("开始重建索引...")

	startTime := time.Now()

	// 选择最优索引类型
	indexType := io.selectOptimalIndexType()
	log.Info("选择索引类型: %s", indexType)

	// 设置索引参数
	indexParams := io.getOptimalIndexParams(indexType)

	// 执行索引重建
	err := io.vectorDB.RebuildIndex(indexType, indexParams)
	if err != nil {
		log.Error("重建索引失败: %v", err)
		return
	}

	// 更新索引统计
	io.mu.Lock()
	io.indexStats.LastRebuildTime = time.Now()
	io.indexStats.BuildTime = time.Since(startTime)
	io.indexStats.DataChanges = 0
	io.mu.Unlock()

	log.Info("索引重建完成，耗时: %v", time.Since(startTime))
}

// incrementalIndex 增量索引
func (io *IndexOptimizer) incrementalIndex() {
	log.Info("开始增量索引...")

	startTime := time.Now()

	// 执行增量索引
	err := io.vectorDB.IncrementalIndex()
	if err != nil {
		log.Error("增量索引失败: %v", err)
		return
	}

	// 更新索引统计
	io.mu.Lock()
	io.indexStats.DataChanges = 0
	io.mu.Unlock()

	log.Info("增量索引完成，耗时: %v", time.Since(startTime))
}

// optimizeIndex 优化索引
func (io *IndexOptimizer) optimizeIndex() {
	log.Info("开始优化索引...")

	startTime := time.Now()

	// 执行索引优化
	err := io.vectorDB.OptimizeIndex()
	if err != nil {
		log.Error("优化索引失败: %v", err)
		return
	}

	log.Info("索引优化完成，耗时: %v", time.Since(startTime))
}

// selectOptimalIndexType 选择最优索引类型
func (io *IndexOptimizer) selectOptimalIndexType() string {
	io.mu.RLock()
	defer io.mu.RUnlock()

	// 根据数据规模和维度选择索引类型
	totalVectors := io.indexStats.TotalVectors
	vectorDim, err := io.vectorDB.GetVectorDimension()

	// 获取硬件信息
	memoryInfo := io.getSystemMemoryInfo()
	cpuCores := runtime.NumCPU()
	hasGPU := io.vectorDB.HasGPUAcceleration()

	// 小规模数据集处理
	if totalVectors < 10000 {
		// 小规模数据集，使用暴力搜索或简单索引
		return "FLAT"
	}

	// 中等规模数据集处理
	if totalVectors < 100000 {
		// 如果维度较高且内存充足，优先使用HNSW
		if vectorDim > 100 && memoryInfo.AvailableGB > 4 {
			return "HNSW"
		}
		// 如果CPU核心较多，可以使用IVF
		if cpuCores >= 8 {
			return "IVF"
		}
		// 默认使用HNSW
		return "HNSW"
	}

	// 大规模数据集处理
	if totalVectors < 1000000 {
		// 如果有GPU加速，优先使用IVF
		if hasGPU {
			return "IVF"
		}
		// 如果维度很高，考虑使用压缩
		if vectorDim > 1000 {
			return "IVFPQ"
		}
		// 如果内存受限，使用压缩
		if memoryInfo.AvailableGB < 8 {
			return "IVFPQ"
		}
		// 默认使用IVF
		return "IVF"
	}

	// 超大规模数据集处理
	// 如果有GPU且内存充足，可以使用IVF
	if hasGPU && memoryInfo.AvailableGB > 32 {
		return "IVF"
	}
	// 默认使用IVF+PQ压缩
	return "IVFPQ"
}

// getSystemMemoryInfo 获取系统内存信息
func (io *IndexOptimizer) getSystemMemoryInfo() struct{ TotalGB, AvailableGB float64 } {
	// 这里是简化实现，实际应使用系统API获取真实内存信息
	// 例如在Linux上可以解析/proc/meminfo，在Windows上可以使用GlobalMemoryStatusEx
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	totalGB := float64(m.TotalAlloc) / (1024 * 1024 * 1024)
	availableGB := totalGB - float64(m.Alloc)/(1024*1024*1024)

	return struct {
		TotalGB, AvailableGB float64
	}{
		TotalGB:     totalGB,
		AvailableGB: availableGB,
	}
}

// getOptimalIndexParams 获取最优索引参数
func (io *IndexOptimizer) getOptimalIndexParams(indexType string) map[string]interface{} {
	params := make(map[string]interface{})

	switch indexType {
	case "HNSW":
		// HNSW参数
		params["m"] = 16                // 每个节点的最大连接数
		params["ef_construction"] = 200 // 构建时的搜索宽度
		params["ef"] = 50               // 搜索时的宽度

	case "IVF":
		// IVF参数
		totalVectors := io.indexStats.TotalVectors
		// 根据数据规模动态调整聚类数
		nlist := int(float64(totalVectors) / 50)
		if nlist < 100 {
			nlist = 100
		} else if nlist > 10000 {
			nlist = 10000
		}
		params["nlist"] = nlist
		params["nprobe"] = nlist / 10 // 默认探测聚类数

	case "IVFPQ":
		// IVF+PQ参数
		totalVectors := io.indexStats.TotalVectors
		nlist := int(float64(totalVectors) / 50)
		if nlist < 100 {
			nlist = 100
		} else if nlist > 10000 {
			nlist = 10000
		}
		params["nlist"] = nlist
		params["nprobe"] = nlist / 10
		params["m"] = 8     // PQ子量化器数量
		params["nbits"] = 8 // 每个子量化器的位数
	}

	return params
}

// RecordDataChange 记录数据变化
func (io *IndexOptimizer) RecordDataChange(count int) {
	io.mu.Lock()
	defer io.mu.Unlock()

	io.indexStats.DataChanges += int64(count)
}

// GetIndexStats 获取索引统计
func (io *IndexOptimizer) GetIndexStats() *IndexStats {
	io.mu.RLock()
	defer io.mu.RUnlock()

	// 返回统计的副本
	statsCopy := *io.indexStats
	return &statsCopy
}

// selectOptimalStrategy 选择最优策略
func (hto *HighThroughputOptimizer) selectOptimalStrategy(data []float64, k int, options *SearchOptions) string {
	// 如果强制使用特定策略，直接返回
	if options.ForceStrategy != "" {
		return options.ForceStrategy
	}

	// 构建搜索上下文
	searchCtx := vector.SearchContext{
		QueryVector:  data,
		K:            k,
		QualityLevel: options.QualityLevel,
		Timeout:      options.Timeout,
	}

	// 使用向量数据库的自适应索引选择器
	strategy := hto.vectorDB.SelectOptimalIndexStrategy(searchCtx)

	// 将内部策略转换为字符串
	strategyStr := "brute_force"
	switch strategy {
	case vector.StrategyBruteForce:
		strategyStr = "brute_force"
	case vector.StrategyIVF:
		strategyStr = "ivf"
	case vector.StrategyHNSW:
		strategyStr = "hnsw"
	case vector.StrategyPQ:
		strategyStr = "pq"
	case vector.StrategyHybrid:
		strategyStr = "hybrid"
	case vector.StrategyEnhancedIVF:
		strategyStr = "enhanced_ivf"
	case vector.StrategyEnhancedLSH:
		strategyStr = "enhanced_lsh"
	}

	return strategyStr
}

// executeSearch 执行搜索
func (hto *HighThroughputOptimizer) executeSearch(ctx context.Context, data []float64, k int, options *SearchOptions, strategy string) ([]entity.Result, error) {
	// 构建搜索上下文
	searchCtx := vector.SearchContext{
		QueryVector:  data,
		K:            k,
		QualityLevel: options.QualityLevel,
		Timeout:      options.Timeout,
	}

	// 根据策略设置搜索参数
	switch strategy {
	case "ivf":
		searchCtx.UseIVF = true
		searchCtx.Nprobe = options.Nprobe
	case "hnsw":
		searchCtx.UseHNSW = true
	case "pq":
		searchCtx.UsePQ = true
	case "hybrid":
		searchCtx.UseHybrid = true
	case "enhanced_ivf":
		searchCtx.UseEnhancedIVF = true
	case "enhanced_lsh":
		searchCtx.UseEnhancedLSH = true
	}

	// 设置GPU选项
	if options.EnableGPU {
		searchCtx.UseGPU = true
	}

	// 创建与SearchContext对应的SearchOptions
	searchOptions := vector.SearchOptions{
		Nprobe:        searchCtx.Nprobe,
		SearchTimeout: searchCtx.Timeout,
		QualityLevel:  searchCtx.QualityLevel,
		UseCache:      options.EnableCache,
		MaxCandidates: k * 2, // 设置一个合理的候选数量
	}

	// 执行优化搜索
	return hto.vectorDB.OptimizedSearch(data, k, searchOptions)
}
