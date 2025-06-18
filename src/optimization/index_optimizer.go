package optimization

import (
	"VectorSphere/src/library/logger"
	"VectorSphere/src/vector"
	"math"
	"math/rand"
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

	// 自适应索引选择相关字段
	performanceWindow []PerformanceRecord
	windowSize        int
	similarityWeights map[string]float64
	lastAdaptiveOpt   time.Time
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

	// 是否启用自适应索引选择
	EnableAdaptiveSelection bool `json:"enable_adaptive_selection"`

	// 性能记录窗口大小
	PerformanceWindowSize int `json:"performance_window_size"`

	// 自适应优化间隔
	AdaptiveOptimizationInterval time.Duration `json:"adaptive_optimization_interval"`

	// 质量权重（在自适应选择中的权重）
	QualityWeight float64 `json:"quality_weight"`

	// 延迟权重（在自适应选择中的权重）
	LatencyWeight float64 `json:"latency_weight"`

	// 内存权重（在自适应选择中的权重）
	MemoryWeight float64 `json:"memory_weight"`
}

// PerformanceRecord 性能记录结构
type PerformanceRecord struct {
	IndexType   string                 // 索引类型
	Latency     time.Duration          // 延迟
	Quality     float64                // 质量
	Timestamp   time.Time              // 时间戳
	VectorCount int                    // 向量数量
	Dimension   int                    // 维度
	MemoryUsage uint64                 // 内存使用
	Params      map[string]interface{} // 索引参数
}

// StrategyPerformance 策略性能统计
type StrategyPerformance struct {
	AvgLatency  time.Duration // 平均延迟
	AvgQuality  float64       // 平均质量
	UsageCount  int           // 使用次数
	LastUsed    time.Time     // 最后使用时间
	SuccessRate float64       // 成功率
	MemoryUsage uint64        // 内存使用
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

	// 各策略性能统计
	StrategyStats map[string]*StrategyPerformance `json:"strategy_stats"`
}

// NewIndexOptimizer 创建索引优化器
func NewIndexOptimizer(vectorDB *vector.VectorDB, config *IndexOptimizerConfig) *IndexOptimizer {
	if config == nil {
		config = getDefaultIndexOptimizerConfig()
	}

	// 初始化相似度权重
	similarityWeights := map[string]float64{
		"vectorCount": 0.4, // 数据规模权重
		"dimension":   0.3, // 维度权重
		"time":        0.3, // 时间权重
	}

	return &IndexOptimizer{
		vectorDB:          vectorDB,
		config:            config,
		lastOptimization:  time.Now(),
		indexStats:        &IndexStats{LastRebuildTime: time.Now(), StrategyStats: make(map[string]*StrategyPerformance)},
		performanceWindow: make([]PerformanceRecord, 0),
		windowSize:        config.PerformanceWindowSize,
		similarityWeights: similarityWeights,
		lastAdaptiveOpt:   time.Now(),
	}
}

// getDefaultIndexOptimizerConfig 获取默认索引优化器配置
func getDefaultIndexOptimizerConfig() *IndexOptimizerConfig {
	return &IndexOptimizerConfig{
		ReindexThreshold:             0.1, // 10%数据变化触发重建
		OptimizationInterval:         1 * time.Hour,
		AutoReindex:                  true,
		IndexOptimization:            true,
		MaxIndexBuildTime:            30 * time.Minute,
		IndexQualityThreshold:        0.9, // 90%质量阈值
		EnableIncrementalIndex:       true,
		IncrementalThreshold:         0.05,            // 5%新增数据触发增量索引
		EnableAdaptiveSelection:      true,            // 启用自适应索引选择
		PerformanceWindowSize:        100,             // 保留最近100次性能记录
		AdaptiveOptimizationInterval: 5 * time.Minute, // 每5分钟优化一次
		QualityWeight:                0.5,             // 质量权重
		LatencyWeight:                0.3,             // 延迟权重
		MemoryWeight:                 0.2,             // 内存权重
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

	// 如果启用了自适应索引选择，启动自适应优化调度器
	if io.config.EnableAdaptiveSelection {
		io.StartAdaptiveOptimization()
	}

	logger.Info("索引优化调度器已启动")
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

	logger.Info("开始索引优化...")

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

	logger.Info("索引优化完成")
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

	// 获取向量数量
	vectorCount := io.vectorDB.GetDataSize()

	// 获取索引健康状态
	indexHealth := io.vectorDB.CheckIndexHealth()

	// 更新索引统计
	io.indexStats.TotalVectors = vectorCount

	// 内存使用情况作为索引大小的近似值
	io.indexStats.IndexSize = int64(dbStats.MemoryUsage)

	// 默认质量值，实际应根据索引类型和参数动态计算
	io.indexStats.Quality = 0.8

	// 获取当前索引类型
	indexType := "unknown"
	if indexHealth["traditional_ivf"] {
		indexType = "IVF"
	} else if indexHealth["hnsw"] {
		indexType = "HNSW"
	} else if indexHealth["enhanced_lsh"] {
		indexType = "EnhancedLSH"
	} else if indexHealth["enhanced_ivf"] {
		indexType = "EnhancedIVF"
	} else if indexHealth["pq"] {
		indexType = "PQ"
	}
	io.indexStats.IndexType = indexType

	// 获取索引参数
	io.indexStats.Parameters = make(map[string]interface{})

	// 根据索引类型设置参数
	if indexType == "IVF" || indexType == "EnhancedIVF" {
		// 为IVF索引设置默认参数，实际应从vectorDB获取
		if io.indexStats.TotalVectors < 10000 {
			io.indexStats.Parameters["nlist"] = 100
		} else if io.indexStats.TotalVectors < 100000 {
			io.indexStats.Parameters["nlist"] = 256
		} else {
			io.indexStats.Parameters["nlist"] = 1024
		}
		io.indexStats.Parameters["nprobe"] = 16
	} else if indexType == "HNSW" {
		// 为HNSW索引设置默认参数
		io.indexStats.Parameters["ef_construction"] = 200
		io.indexStats.Parameters["m"] = 16
	} else if indexType == "EnhancedLSH" {
		// 为LSH索引设置默认参数
		io.indexStats.Parameters["num_hash_tables"] = 10
		io.indexStats.Parameters["num_hash_functions"] = 8
	} else if indexType == "PQ" {
		// 为PQ索引设置默认参数
		io.indexStats.Parameters["num_sub_vectors"] = 8
		io.indexStats.Parameters["num_centroids"] = 256
	}

	logger.Info("索引统计: 类型=%s, 大小=%dMB, 质量=%.2f, 向量数=%d",
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
		logger.Info("数据变化比例(%.2f%%)超过阈值(%.2f%%)，需要重建索引",
			changeRatio*100, io.config.ReindexThreshold*100)
		return true
	}

	// 检查索引质量
	if io.indexStats.Quality < io.config.IndexQualityThreshold {
		logger.Info("索引质量(%.2f)低于阈值(%.2f)，需要重建索引",
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
		logger.Info("数据变化比例(%.2f%%)适合增量索引", changeRatio*100)
		return true
	}

	return false
}

// rebuildIndex 重建索引
func (io *IndexOptimizer) rebuildIndex() {
	logger.Info("开始重建索引...")

	startTime := time.Now()

	// 选择最优索引类型
	indexType := io.selectOptimalIndexType()
	logger.Info("选择索引类型: %s", indexType)

	// 设置索引参数
	indexParams := io.getOptimalIndexParams(indexType)

	// 重建索引
	logger.Info("开始重建索引，类型: %s, 参数: %v", indexType, indexParams)
	err := io.vectorDB.RebuildIndexWithType(indexType, indexParams) // 使用 indexType 和 indexParams 参数
	if err != nil {
		logger.Error("重建索引失败: %v", err)
	}

	// 计算重建耗时和质量
	buildTime := time.Since(startTime)
	// 估算索引质量（可以通过采样测试或使用默认值）
	quality := 0.95 // 默认假设重建后的索引质量较高

	// 记录索引性能
	if io.config.EnableAdaptiveSelection {
		io.recordIndexPerformance(indexType, buildTime, quality, indexParams)
	}

	// 更新索引统计
	io.mu.Lock()
	io.indexStats.LastRebuildTime = time.Now()
	io.indexStats.BuildTime = buildTime
	io.indexStats.DataChanges = 0
	io.indexStats.IndexType = indexType
	io.indexStats.Parameters = indexParams
	io.mu.Unlock()

	logger.Info("索引重建完成，耗时: %v", buildTime)
}

// incrementalIndex 增量索引
func (io *IndexOptimizer) incrementalIndex() {
	logger.Info("开始增量索引...")

	startTime := time.Now()

	// 执行增量索引
	err := io.vectorDB.IncrementalIndex()
	if err != nil {
		logger.Error("增量索引失败: %v", err)
		return
	}

	// 更新索引统计
	io.mu.Lock()
	io.indexStats.DataChanges = 0
	io.mu.Unlock()

	logger.Info("增量索引完成，耗时: %v", time.Since(startTime))
}

// optimizeIndex 优化索引
func (io *IndexOptimizer) optimizeIndex() {
	logger.Info("开始优化索引...")

	startTime := time.Now()

	// 执行索引优化
	err := io.vectorDB.OptimizeIndex()
	if err != nil {
		logger.Error("优化索引失败: %v", err)
		return
	}

	logger.Info("索引优化完成，耗时: %v", time.Since(startTime))
}

// selectOptimalIndexType 选择最优索引类型
func (io *IndexOptimizer) selectOptimalIndexType() string {
	io.mu.RLock()
	defer io.mu.RUnlock()

	// 如果启用了自适应索引选择，优先使用自适应方法
	if io.config.EnableAdaptiveSelection && len(io.performanceWindow) >= 5 {
		if indexType := io.selectAdaptiveIndexType(); indexType != "" {
			logger.Info("使用自适应索引选择: %s", indexType)
			return indexType
		}
	}

	// 根据数据规模和维度选择索引类型
	totalVectors := io.indexStats.TotalVectors
	vectorDim, err := io.vectorDB.GetVectorDimension()

	// 获取硬件信息
	memoryInfo := io.getSystemMemoryInfo()
	//cpuCores := runtime.NumCPU()
	hasGPU := io.vectorDB.HardwareCaps.HasGPU

	// 检查增强型索引是否可用
	indexHealth := io.vectorDB.CheckIndexHealth()
	ivfIndexReady := indexHealth["enhanced_ivf"]
	lshIndexReady := indexHealth["enhanced_lsh"]

	// 小规模数据集处理
	if totalVectors < 1000 {
		return "FLAT"
	}

	// 中等规模数据集处理
	if totalVectors < 10000 {
		// 对于中等规模，优先考虑HNSW
		return "HNSW"
	}

	// 大规模数据集处理
	if err == nil {
		// 高维数据优先考虑LSH
		if vectorDim > 1000 {
			if lshIndexReady {
				return "EnhancedLSH"
			}
			return "LSH"
		}

		// 中低维数据优先考虑IVF
		if vectorDim <= 1000 {
			if ivfIndexReady {
				return "EnhancedIVF"
			}
			return "IVF"
		}
	}

	// 超大规模数据集处理
	if totalVectors > 1000000 {
		// 如果内存受限，优先考虑PQ压缩
		if memoryInfo.TotalGB < 8*1024*1024*1024 { // 小于8GB内存
			return "IVFPQ"
		}

		// 如果有GPU加速，优先考虑增强型IVF
		if hasGPU && ivfIndexReady {
			return "EnhancedIVF"
		}

		return "IVF"
	}

	// 默认返回IVF
	return "IVF"
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

	// 如果启用了自适应索引选择，尝试从历史记录中获取最佳参数
	if io.config.EnableAdaptiveSelection && len(io.performanceWindow) >= 5 {
		if bestParams := io.getBestHistoricalParams(indexType); len(bestParams) > 0 {
			logger.Info("使用历史最佳参数配置: %v", bestParams)
			return bestParams
		}
	}

	// 获取向量维度
	vectorDim, _ := io.vectorDB.GetVectorDimension()
	// 获取硬件信息
	memoryInfo := io.getSystemMemoryInfo()
	//cpuCores := runtime.NumCPU()
	hasGPU := io.vectorDB.HardwareCaps.HasGPU
	totalVectors := io.indexStats.TotalVectors

	switch indexType {
	case "HNSW":
		// HNSW参数 - 根据数据规模和维度动态调整
		m := 16 // 默认每个节点的最大连接数

		// 根据维度调整M值
		if vectorDim > 500 {
			m = 24 // 高维度需要更多连接
		} else if vectorDim < 50 {
			m = 12 // 低维度可以减少连接
		}

		// 根据数据规模调整ef_construction
		efConstruction := 200 // 默认构建时的搜索宽度
		if totalVectors > 500000 {
			efConstruction = 400 // 大规模数据需要更高的构建精度
		} else if totalVectors < 10000 {
			efConstruction = 100 // 小规模数据可以降低构建精度
		}

		// 根据质量要求调整ef
		ef := 50 // 默认搜索时的宽度
		if io.config.IndexQualityThreshold > 0.95 {
			ef = 100 // 高质量要求需要更大的搜索宽度
		} else if io.config.IndexQualityThreshold < 0.8 {
			ef = 30 // 低质量要求可以减少搜索宽度
		}

		params["maxConnections"] = m
		params["efConstruction"] = efConstruction
		params["efSearch"] = ef

	case "IVF":
		// IVF参数 - 根据数据规模动态调整聚类数
		nlist := int(float64(totalVectors) / 50)
		if nlist < 100 {
			nlist = 100
		} else if nlist > 10000 {
			nlist = 10000
		}

		// 根据质量要求调整nprobe
		nprobe := nlist / 10 // 默认探测聚类数
		if io.config.IndexQualityThreshold > 0.95 {
			nprobe = nlist / 5 // 高质量要求需要探测更多聚类
		} else if io.config.IndexQualityThreshold < 0.8 {
			nprobe = nlist / 20 // 低质量要求可以减少探测聚类
		}

		params["nlist"] = nlist
		params["nprobe"] = nprobe

	case "IVFPQ":
		// IVF+PQ参数 - 根据数据规模和维度动态调整
		nlist := int(float64(totalVectors) / 50)
		if nlist < 100 {
			nlist = 100
		} else if nlist > 10000 {
			nlist = 10000
		}

		// 根据维度调整PQ子量化器数量
		m := 8 // 默认PQ子量化器数量
		if vectorDim > 500 {
			m = 16 // 高维度需要更多子量化器
		} else if vectorDim < 100 {
			m = 4 // 低维度可以减少子量化器
		}

		// 根据内存限制调整nbits
		nbits := 8 // 默认每个子量化器的位数
		if memoryInfo.AvailableGB < 4 {
			nbits = 4 // 内存受限时减少位数
		} else if memoryInfo.AvailableGB > 32 {
			nbits = 12 // 内存充足时增加位数
		}

		params["nlist"] = nlist
		params["nprobe"] = nlist / 10
		params["m"] = m
		params["nbits"] = nbits

	case "EnhancedIVF":
		// 增强型IVF参数
		nlist := int(float64(totalVectors) / 40) // 比普通IVF更精细的聚类
		if nlist < 200 {
			nlist = 200
		} else if nlist > 20000 {
			nlist = 20000
		}

		// 根据质量要求调整nprobe
		nprobe := nlist / 8 // 默认探测聚类数
		if io.config.IndexQualityThreshold > 0.95 {
			nprobe = nlist / 4 // 高质量要求需要探测更多聚类
		} else if io.config.IndexQualityThreshold < 0.8 {
			nprobe = nlist / 16 // 低质量要求可以减少探测聚类
		}

		params["nlist"] = nlist
		params["nprobe"] = nprobe
		params["use_residual"] = true    // 使用残差量化
		params["encode_residual"] = true // 编码残差

		// 如果有GPU加速，启用GPU索引
		if hasGPU {
			params["use_gpu"] = true
			params["gpu_id"] = 0 // 默认使用第一个GPU
		}

	case "EnhancedLSH":
		// 增强型LSH参数
		// 根据维度调整哈希函数数量
		numHashFunctions := 8 // 默认哈希函数数量
		if vectorDim > 500 {
			numHashFunctions = 16 // 高维度需要更多哈希函数
		} else if vectorDim < 100 {
			numHashFunctions = 4 // 低维度可以减少哈希函数
		}

		// 根据数据规模调整哈希表数量
		numHashTables := 10 // 默认哈希表数量
		if totalVectors > 500000 {
			numHashTables = 20 // 大规模数据需要更多哈希表
		} else if totalVectors < 10000 {
			numHashTables = 5 // 小规模数据可以减少哈希表
		}

		params["num_hash_functions"] = numHashFunctions
		params["num_hash_tables"] = numHashTables
		params["bucket_size"] = 256  // 默认桶大小
		params["multi_probe"] = true // 启用多探测
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

// 记录索引性能
func (io *IndexOptimizer) recordIndexPerformance(indexType string, latency time.Duration, quality float64, params map[string]interface{}) {
	io.mu.Lock()
	defer io.mu.Unlock()

	// 获取向量维度
	vectorDim, _ := io.vectorDB.GetVectorDimension()

	// 获取内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// 创建性能记录
	record := PerformanceRecord{
		IndexType:   indexType,
		Latency:     latency,
		Quality:     quality,
		Timestamp:   time.Now(),
		VectorCount: io.indexStats.TotalVectors,
		Dimension:   vectorDim,
		MemoryUsage: m.Alloc,
		Params:      params,
	}

	// 添加到性能窗口
	io.performanceWindow = append(io.performanceWindow, record)

	// 保持窗口大小
	if len(io.performanceWindow) > io.windowSize {
		io.performanceWindow = io.performanceWindow[1:]
	}

	// 更新索引统计中的策略性能
	if io.indexStats.StrategyStats == nil {
		io.indexStats.StrategyStats = make(map[string]*StrategyPerformance)
	}

	// 获取或创建策略性能统计
	stats, exists := io.indexStats.StrategyStats[indexType]
	if !exists {
		stats = &StrategyPerformance{
			AvgLatency:  latency,
			AvgQuality:  quality,
			UsageCount:  1,
			LastUsed:    time.Now(),
			SuccessRate: 1.0,
			MemoryUsage: m.Alloc,
		}
		io.indexStats.StrategyStats[indexType] = stats
	} else {
		// 更新平均值（使用指数移动平均）
		alpha := 0.1 // 平滑因子
		stats.AvgLatency = time.Duration(float64(stats.AvgLatency)*(1-alpha) + float64(latency)*alpha)
		stats.AvgQuality = stats.AvgQuality*(1-alpha) + quality*alpha
		stats.UsageCount++
		stats.LastUsed = time.Now()
		stats.MemoryUsage = uint64(float64(stats.MemoryUsage)*(1-alpha) + float64(m.Alloc)*alpha)
	}

	logger.Info("记录索引性能: 类型=%s, 延迟=%v, 质量=%.2f, 向量数=%d, 维度=%d",
		indexType, latency, quality, io.indexStats.TotalVectors, vectorDim)
}

// 选择自适应索引类型
func (io *IndexOptimizer) selectAdaptiveIndexType() string {
	if len(io.performanceWindow) < 5 {
		return "" // 数据不足，返回空字符串
	}

	// 获取当前上下文
	vectorCount := io.indexStats.TotalVectors
	vectorDim, _ := io.vectorDB.GetVectorDimension()

	// 计算各索引类型的综合评分
	typeScores := make(map[string]float64)
	typeCount := make(map[string]int)

	for _, record := range io.performanceWindow {
		// 计算上下文相似度
		contextSimilarity := io.calculateContextSimilarity(record, vectorCount, vectorDim)

		// 计算性能评分 = 质量权重 * 质量 + 延迟权重 * (1 / 延迟) + 内存权重 * (1 / 内存使用)
		qualityScore := io.config.QualityWeight * record.Quality
		latencyScore := io.config.LatencyWeight * (1.0 / (float64(record.Latency) / float64(time.Millisecond)))
		memoryScore := io.config.MemoryWeight * (1.0 / (float64(record.MemoryUsage) / (1024 * 1024 * 1024)))

		// 归一化延迟和内存评分
		latencyScore = math.Min(1.0, latencyScore/1000) // 假设1ms是最佳延迟
		memoryScore = math.Min(1.0, memoryScore*10)     // 假设100MB是最佳内存使用

		// 综合评分 = 上下文相似度 * (质量评分 + 延迟评分 + 内存评分)
		score := contextSimilarity * (qualityScore + latencyScore + memoryScore)

		// 累加评分
		typeScores[record.IndexType] += score
		typeCount[record.IndexType]++
	}

	// 找出评分最高的索引类型
	bestType := ""
	bestScore := -1.0

	for indexType, totalScore := range typeScores {
		// 计算平均评分
		avgScore := totalScore / float64(typeCount[indexType])

		if avgScore > bestScore {
			bestScore = avgScore
			bestType = indexType
		}
	}

	return bestType
}

// 计算上下文相似度
func (io *IndexOptimizer) calculateContextSimilarity(record PerformanceRecord, currentVectorCount, currentDimension int) float64 {
	// 数据规模相似度
	vectorCountSim := 1.0 - math.Abs(float64(currentVectorCount-record.VectorCount))/math.Max(float64(currentVectorCount), float64(record.VectorCount))

	// 维度相似度
	dimensionSim := 1.0 - math.Abs(float64(currentDimension-record.Dimension))/math.Max(float64(currentDimension), float64(record.Dimension))

	// 时间相似度 (越近的记录相似度越高)
	timeDiff := time.Since(record.Timestamp).Hours()
	timeSim := math.Exp(-timeDiff / 24.0) // 24小时衰减

	// 综合相似度 (加权平均)
	similarity := io.similarityWeights["vectorCount"]*vectorCountSim +
		io.similarityWeights["dimension"]*dimensionSim +
		io.similarityWeights["time"]*timeSim

	return math.Max(0, math.Min(1, similarity))
}

// 获取历史最佳参数
func (io *IndexOptimizer) getBestHistoricalParams(indexType string) map[string]interface{} {
	if len(io.performanceWindow) < 5 {
		return nil // 数据不足
	}

	// 获取当前上下文
	vectorCount := io.indexStats.TotalVectors
	vectorDim, _ := io.vectorDB.GetVectorDimension()

	// 筛选出指定索引类型的记录
	typeRecords := make([]PerformanceRecord, 0)
	for _, record := range io.performanceWindow {
		if record.IndexType == indexType {
			typeRecords = append(typeRecords, record)
		}
	}

	if len(typeRecords) < 3 {
		return nil // 该类型的数据不足
	}

	// 计算每条记录的综合评分
	bestRecord := typeRecords[0]
	bestScore := -1.0

	for _, record := range typeRecords {
		// 计算上下文相似度
		contextSimilarity := io.calculateContextSimilarity(record, vectorCount, vectorDim)

		// 计算性能评分
		qualityScore := io.config.QualityWeight * record.Quality
		latencyScore := io.config.LatencyWeight * (1.0 / (float64(record.Latency) / float64(time.Millisecond)))

		// 归一化延迟评分
		latencyScore = math.Min(1.0, latencyScore/1000) // 假设1ms是最佳延迟

		// 综合评分 = 上下文相似度 * (质量评分 + 延迟评分)
		score := contextSimilarity * (qualityScore + latencyScore)

		if score > bestScore {
			bestScore = score
			bestRecord = record
		}
	}

	return bestRecord.Params
}

// 优化自适应参数
func (io *IndexOptimizer) optimizeAdaptiveParams() {
	io.mu.Lock()
	defer io.mu.Unlock()

	if len(io.performanceWindow) < 10 {
		return // 数据不足，无法优化
	}

	// 分析不同索引类型的性能表现
	typePerf := make(map[string][]PerformanceRecord)
	for _, record := range io.performanceWindow {
		typePerf[record.IndexType] = append(typePerf[record.IndexType], record)
	}

	// 优化相似度权重
	bestWeights := io.similarityWeights
	bestScore := io.evaluateWeights(io.similarityWeights, typePerf)

	// 尝试不同的权重组合
	for i := 0; i < 10; i++ {
		// 随机生成权重
		weights := map[string]float64{
			"vectorCount": 0.2 + 0.6*rand.Float64(),
			"dimension":   0.1 + 0.4*rand.Float64(),
			"time":        0.1 + 0.4*rand.Float64(),
		}

		// 归一化权重
		total := weights["vectorCount"] + weights["dimension"] + weights["time"]
		for k := range weights {
			weights[k] /= total
		}

		// 评估权重
		score := io.evaluateWeights(weights, typePerf)

		if score > bestScore {
			bestScore = score
			bestWeights = weights
		}
	}

	// 更新最佳权重
	io.similarityWeights = bestWeights
	logger.Info("优化自适应参数: 向量数权重=%.2f, 维度权重=%.2f, 时间权重=%.2f",
		io.similarityWeights["vectorCount"], io.similarityWeights["dimension"], io.similarityWeights["time"])
}

// 评估权重
func (io *IndexOptimizer) evaluateWeights(weights map[string]float64, typePerf map[string][]PerformanceRecord) float64 {
	// 模拟一些查询场景
	scenarios := []struct {
		vectorCount  int
		dimension    int
		timestamp    time.Time
		expectedType string
	}{
		{1000, 128, time.Now().Add(-24 * time.Hour), "FLAT"},
		{50000, 256, time.Now().Add(-12 * time.Hour), "HNSW"},
		{500000, 512, time.Now().Add(-6 * time.Hour), "IVF"},
		{2000000, 1024, time.Now().Add(-1 * time.Hour), "IVFPQ"},
	}

	correctPredictions := 0

	for _, scenario := range scenarios {
		// 创建模拟记录
		//mockRecord := PerformanceRecord{
		//	VectorCount: scenario.vectorCount,
		//	Dimension:   scenario.dimension,
		//	Timestamp:   scenario.timestamp,
		//}

		// 找出最相似的索引类型
		bestType := ""
		bestSimilarity := -1.0

		for indexType, records := range typePerf {
			for _, record := range records {
				// 计算上下文相似度（使用评估的权重）
				vectorCountSim := 1.0 - math.Abs(float64(scenario.vectorCount-record.VectorCount))/math.Max(float64(scenario.vectorCount), float64(record.VectorCount))
				dimensionSim := 1.0 - math.Abs(float64(scenario.dimension-record.Dimension))/math.Max(float64(scenario.dimension), float64(record.Dimension))
				timeDiff := scenario.timestamp.Sub(record.Timestamp).Hours()
				timeSim := math.Exp(-math.Abs(timeDiff) / 24.0) // 24小时衰减

				// 综合相似度 (加权平均)
				similarity := weights["vectorCount"]*vectorCountSim +
					weights["dimension"]*dimensionSim +
					weights["time"]*timeSim

				if similarity > bestSimilarity {
					bestSimilarity = similarity
					bestType = indexType
				}
			}
		}

		// 检查预测是否正确
		if bestType == scenario.expectedType {
			correctPredictions++
		}
	}

	// 返回准确率
	return float64(correctPredictions) / float64(len(scenarios))
}

// StartAdaptiveOptimization 启动自适应优化
func (io *IndexOptimizer) StartAdaptiveOptimization() {
	if !io.config.EnableAdaptiveSelection {
		return
	}

	go func() {
		ticker := time.NewTicker(io.config.AdaptiveOptimizationInterval)
		defer ticker.Stop()

		for range ticker.C {
			io.optimizeAdaptiveParams()
		}
	}()

	logger.Info("自适应优化调度器已启动，间隔: %v", io.config.AdaptiveOptimizationInterval)
}
