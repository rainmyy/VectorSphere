package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/algorithm"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/enum"
	"VectorSphere/src/library/graph"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/library/tree"
	"container/heap"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// IndexStrategy 索引策略枚举
type IndexStrategy int

const (
	StrategyBruteForce IndexStrategy = iota // 暴力搜索
	StrategyIVF                             // IVF索引
	StrategyHNSW                            // HNSW索引
	StrategyPQ                              // PQ压缩搜索
	StrategyHybrid                          // 混合策略
	StrategyEnhancedIVF
	StrategyEnhancedLSH
	StrategyIVFHNSW // IVF-HNSW混合索引
)

// SearchContext 搜索上下文
type SearchContext struct {
	QueryVector  []float64
	K            int
	Nprobe       int
	Timeout      time.Duration
	QualityLevel float64 // 0.0-1.0，质量要求等级
	// 索引策略选项
	UseIVF         bool
	UseHNSW        bool
	UsePQ          bool
	UseHybrid      bool
	UseEnhancedIVF bool
	UseEnhancedLSH bool
	UseGPU         bool
}

// StrategySelector 智能策略选择器
type StrategySelector struct {
	db          *VectorDB
	performance map[IndexStrategy]*PerformanceMetrics
	mu          sync.RWMutex
}

// PerformanceMetrics 性能指标
type PerformanceMetrics struct {
	AvgLatency    time.Duration
	Recall        float64
	ThroughputQPS float64
	MemoryUsage   uint64
	LastUpdated   time.Time
}

// Cluster 代表一个向量簇
type Cluster struct {
	Centroid  entity.Point // 簇的中心点
	VectorIDs []string     // 属于该簇的向量ID列表
}

// PerformanceStats 性能统计结构
type PerformanceStats struct {
	TotalQueries    int64
	CacheHits       int64
	AvgQueryTime    time.Duration
	IndexBuildTime  time.Duration
	LastReindexTime time.Time
	MemoryUsage     uint64

	// GPU相关统计
	GPUComputations    int64         // GPU计算次数
	GPUSearches        int64         // GPU搜索次数
	GPUErrors          int64         // GPU错误次数
	TotalGPUTime       int64         // GPU总计算时间(毫秒)
	TotalGPUSearchTime int64         // GPU总搜索时间(毫秒)
	AvgGPUTime         time.Duration // GPU平均计算时间
	GPUUtilization     float64       // GPU利用率
}

type VectorDB struct {
	vectors       map[string][]float64
	mu            sync.RWMutex
	filePath      string              // 数据库文件的存储路径
	backupPath    string              //数据备份逻辑
	clusters      []Cluster           // 存储簇信息，用于IVF索引
	numClusters   int                 // K-Means中的K值，即簇的数量
	indexed       bool                // 标记数据库是否已建立索引
	invertedIndex map[string][]string // 倒排索引，关键词 -> 文件ID列表
	// 添加倒排索引锁，细化锁粒度
	invertedMu sync.RWMutex

	// 距离计算器
	distanceCalculator algorithm.DistanceCalculator // 用于计算向量距离的通用组件

	// 新增字段
	vectorDim         int                  // 向量维度，用于验证
	vectorizedType    int                  // 默认向量化类型
	normalizedVectors map[string][]float64 // 预计算的归一化向量

	compressedVectors map[string]entity.CompressedVector // 压缩后的向量
	useCompression    bool                               // 是否使用压缩
	stats             PerformanceStats
	statsMu           sync.RWMutex
	multiIndex        *MultiLevelIndex // 存储构建的多级索引
	config            AdaptiveConfig

	// 新增 PQ 相关字段
	pqCodebook               [][]entity.Point // 从文件加载的 PQ 码本
	pqCodebookFilePath       string           // PQ 码本文件路径，用于热加载
	numSubVectors            int              // PQ 的子向量数量
	numCentroidsPerSubVector int              // 每个子向量空间的质心数量
	usePQCompression         bool             // 标志是否启用 PQ 压缩

	stopCh           chan struct{}
	useNormalization bool
	hnsw             *graph.HNSWGraph // HNSW 图结构索引
	useHNSWIndex     bool             // 是否使用 HNSW 索引
	maxConnections   int              // HNSW 最大连接数
	efConstruction   float64          // HNSW 构建时的扩展因子
	efSearch         float64          // HNSW 搜索时的扩展因子
	metadata         map[string]map[string]interface{}

	MultiCache *MultiLevelCache // 多级缓存

	// 新增硬件自适应相关字段
	strategyComputeSelector *acceler.ComputeStrategySelector
	currentStrategy         acceler.ComputeStrategy
	HardwareCaps            acceler.HardwareCapabilities
	strategySelector        *StrategySelector

	// 新增 mmap 相关字段
	useMmap     bool // 是否启用 mmap
	mmapEnabled bool // mmap 是否可用

	// 增强 IVF 相关字段
	ivfConfig         *IVFConfig         // IVF 配置
	ivfIndex          *EnhancedIVFIndex  // 增强 IVF 索引
	ivfPQIndex        *IVFPQIndex        // IVF-PQ 混合索引
	dynamicClusters   bool               // 是否启用动态聚类
	clusterUpdateChan chan ClusterUpdate // 聚类更新通道

	// 增强 LSH 相关字段
	LshConfig   *LSHConfig            `json:"lsh_config"`
	LshIndex    *EnhancedLSHIndex     `json:"lsh_index"`
	LshFamilies map[string]*LSHFamily `json:"lsh_families"`
	AdaptiveLSH *AdaptiveLSH          `json:"adaptive_lsh"`

	// IVF-HNSW 混合索引相关字段
	ivfHnswIndex    *IVFHNSWIndex  // IVF-HNSW 混合索引
	ivfHnswConfig   *IVFHNSWConfig // IVF-HNSW 配置
	useIVFHNSWIndex bool           // 是否启用 IVF-HNSW 混合索引

	adaptiveSelector   *AdaptiveIndexSelector // 自适应索引选择器
	cachePath          string
	pcaConfig          *PCAConfig
	performanceMonitor PerformanceMonitor       // 性能监控器
	dataPreprocessor   DataPreprocessor         // 数据预处理器
	hardwareManager    *acceler.HardwareManager // 硬件管理器
	cacheManager       *CacheManager            // 缓存管理器
}

const (
	DefaultVectorized = iota
	SimpleVectorized
	TfidfVectorized
	WordEmbeddingVectorized
)

// IVFHNSWConfig IVF-HNSW 混合索引配置
type IVFHNSWConfig struct {
	// IVF 相关配置
	NumClusters        int     `json:"num_clusters"`        // 聚类数量
	TrainingRatio      float64 `json:"training_ratio"`      // 训练数据比例
	Nprobe             int     `json:"nprobe"`              // 搜索时探测的聚类数量
	RebalanceThreshold int     `json:"rebalance_threshold"` // 重平衡阈值

	// HNSW 相关配置
	MaxConnections int     `json:"max_connections"` // HNSW 最大连接数
	EfConstruction float64 `json:"ef_construction"` // HNSW 构建时的扩展因子
	EfSearch       float64 `json:"ef_search"`       // HNSW 搜索时的扩展因子
	MaxLevel       int     `json:"max_level"`       // HNSW 最大层数

	// 混合索引配置
	EnableHierarchical bool    `json:"enable_hierarchical"` // 启用层次化搜索
	ClusterHNSWRatio   float64 `json:"cluster_hnsw_ratio"`  // 聚类内使用HNSW的比例阈值
	MinClusterSize     int     `json:"min_cluster_size"`    // 使用HNSW的最小聚类大小
	MaxClusterSize     int     `json:"max_cluster_size"`    // 最大聚类大小
	UsePQCompression   bool    `json:"use_pq_compression"`  // 是否使用PQ压缩
	PQSubVectors       int     `json:"pq_sub_vectors"`      // PQ 子向量数量
	PQCentroids        int     `json:"pq_centroids"`        // PQ 质心数量
}

// IVFHNSWIndex IVF-HNSW 混合索引结构
type IVFHNSWIndex struct {
	// IVF 部分
	Clusters         []IVFHNSWCluster `json:"clusters"`
	ClusterCentroids [][]float64      `json:"centroids"`
	ClusterSizes     []int            `json:"sizes"`
	NumClusters      int              `json:"num_clusters"`

	// HNSW 部分
	ClusterGraphs map[int]*graph.HNSWGraph `json:"-"` // 每个聚类的HNSW图
	GlobalGraph   *graph.HNSWGraph         `json:"-"` // 全局HNSW图（聚类中心）

	// 统计信息
	TotalVectors     int                     `json:"total_vectors"`
	IndexVersion     int                     `json:"version"`
	LastUpdateTime   time.Time               `json:"last_update"`
	PerformanceStats IVFHNSWPerformanceStats `json:"performance_stats"`

	// 并发控制
	mu     sync.RWMutex `json:"-"`
	Enable bool         `json:"enable"`
}

// IVFHNSWCluster IVF-HNSW 聚类结构
type IVFHNSWCluster struct {
	Centroid     []float64             `json:"centroid"`
	VectorIDs    []string              `json:"vector_ids"`
	HNSWGraph    *graph.HNSWGraph      `json:"-"` // 聚类内的HNSW图
	UseHNSW      bool                  `json:"use_hnsw"`
	PQCodes      map[string][]byte     `json:"pq_codes,omitempty"` // PQ编码（可选）
	Metrics      IVFHNSWClusterMetrics `json:"metrics"`
	LastAccessed time.Time             `json:"last_accessed"`
	AccessCount  int64                 `json:"access_count"`
}

// IVFHNSWClusterMetrics 聚类指标
type IVFHNSWClusterMetrics struct {
	Variance       float64       `json:"variance"`        // 方差
	Density        float64       `json:"density"`         // 密度
	Radius         float64       `json:"radius"`          // 半径
	QueryFrequency float64       `json:"query_frequency"` // 查询频率
	HNSWBuildTime  time.Duration `json:"hnsw_build_time"` // HNSW构建时间
	AvgSearchTime  time.Duration `json:"avg_search_time"` // 平均搜索时间
	LastRebalance  time.Time     `json:"last_rebalance"`
}

// IVFHNSWPerformanceStats IVF-HNSW 性能统计
type IVFHNSWPerformanceStats struct {
	TotalQueries     int64         `json:"total_queries"`
	IVFQueries       int64         `json:"ivf_queries"`    // 仅使用IVF的查询数
	HNSWQueries      int64         `json:"hnsw_queries"`   // 使用HNSW的查询数
	HybridQueries    int64         `json:"hybrid_queries"` // 混合查询数
	AvgIVFLatency    time.Duration `json:"avg_ivf_latency"`
	AvgHNSWLatency   time.Duration `json:"avg_hnsw_latency"`
	AvgHybridLatency time.Duration `json:"avg_hybrid_latency"`
	Recall           float64       `json:"recall"`
	ThroughputQPS    float64       `json:"throughput_qps"`
	MemoryUsage      uint64        `json:"memory_usage"`
	LastUpdated      time.Time     `json:"last_updated"`
}

// GetStats 获取性能统计信息
func (db *VectorDB) GetStats() PerformanceStats {
	db.statsMu.RLock()
	defer db.statsMu.RUnlock()

	// 计算GPU平均计算时间
	if db.stats.GPUComputations > 0 {
		db.stats.AvgGPUTime = time.Duration(db.stats.TotalGPUTime / db.stats.GPUComputations)
	}

	// 计算GPU利用率（GPU计算次数占总查询的比例）
	if db.stats.TotalQueries > 0 {
		db.stats.GPUUtilization = float64(db.stats.GPUComputations+db.stats.GPUSearches) / float64(db.stats.TotalQueries) * 100
	}

	return db.stats
}

// GetGPUAccelerationStatus 获取GPU加速状态
func (db *VectorDB) GetGPUAccelerationStatus() string {
	// 检查硬件管理器是否初始化
	if db.hardwareManager == nil {
		return "不可用 - 硬件管理器未初始化"
	}

	// 检查GPU是否可用
	if !db.HardwareCaps.HasGPU {
		return "不可用 - 硬件不支持"
	}

	// 通过硬件管理器获取GPU加速器
	gpuAccelerator, ok := db.hardwareManager.GetAccelerator("GPU")
	if !ok {
		return "不可用 - 未初始化"
	}

	// 检查GPU加速器是否可用
	if gpuAccel, ok := gpuAccelerator.(*acceler.FAISSAccelerator); ok {
		if err := gpuAccel.CheckGPUAvailability(); err != nil {
			return fmt.Sprintf("不可用 - %v", err)
		}
		return "可用 - 正常运行"
	}

	return "未知状态"
}

// GetGPUStats 获取GPU加速相关的详细统计信息
func (db *VectorDB) GetGPUStats() map[string]interface{} {
	db.statsMu.RLock()
	defer db.statsMu.RUnlock()

	// 计算GPU平均计算时间
	avgGPUTime := time.Duration(0)
	if db.stats.GPUComputations > 0 {
		avgGPUTime = time.Duration(db.stats.TotalGPUTime / db.stats.GPUComputations)
	}

	// 计算GPU平均搜索时间
	avgGPUSearchTime := time.Duration(0)
	if db.stats.GPUSearches > 0 {
		avgGPUSearchTime = time.Duration(db.stats.TotalGPUSearchTime / db.stats.GPUSearches)
	}

	// 计算GPU利用率
	gpuUtilization := 0.0
	if db.stats.TotalQueries > 0 {
		gpuUtilization = float64(db.stats.GPUComputations+db.stats.GPUSearches) / float64(db.stats.TotalQueries) * 100
	}

	// 计算GPU错误率
	gpuErrorRate := 0.0
	totalGPUOperations := db.stats.GPUComputations + db.stats.GPUSearches
	if totalGPUOperations > 0 {
		gpuErrorRate = float64(db.stats.GPUErrors) / float64(totalGPUOperations) * 100
	}

	// 获取GPU硬件信息
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	gpuHardwareInfo := map[string]interface{}{
		"available":   db.HardwareCaps.HasGPU,
		"initialized": gpuAccelerator != nil,
	}

	// 如果有硬件管理器，获取更详细的GPU配置
	if db.hardwareManager != nil {
		config := db.hardwareManager.GetConfig()
		if config != nil {
			gpuHardwareInfo["config"] = map[string]interface{}{
				"enabled":      config.GPU.Enable,
				"device_id":    config.GPU.DeviceID,
				"memory_limit": config.GPU.MemoryLimit,
				"batch_size":   config.GPU.BatchSize,
			}
		}
	}

	// 构建详细的GPU统计信息
	return map[string]interface{}{
		"computations":             db.stats.GPUComputations,
		"searches":                 db.stats.GPUSearches,
		"errors":                   db.stats.GPUErrors,
		"total_gpu_time_ms":        db.stats.TotalGPUTime,
		"total_gpu_search_time_ms": db.stats.TotalGPUSearchTime,
		"avg_gpu_time":             fmt.Sprintf("%v", avgGPUTime),
		"avg_gpu_search_time":      fmt.Sprintf("%v", avgGPUSearchTime),
		"gpu_utilization":          fmt.Sprintf("%.2f%%", gpuUtilization),
		"gpu_error_rate":           fmt.Sprintf("%.2f%%", gpuErrorRate),
		"hardware_info":            gpuHardwareInfo,
		"acceleration_status":      db.GetGPUAccelerationStatus(),
		"memory_usage":             fmt.Sprintf("%.2f%%", db.stats.GPUUtilization),
	}
}

// SelectOptimalIndexStrategy 智能选择最优索引策略
func (db *VectorDB) SelectOptimalIndexStrategy(ctx SearchContext) IndexStrategy {
	db.mu.RLock()
	vectorCount := len(db.vectors)
	vectorDim := len(ctx.QueryVector)
	ivfIndexReady := db.ivfIndex != nil && db.ivfIndex.Enable
	lshIndexReady := db.LshIndex != nil && db.LshIndex.Enable
	ivfHnswIndexReady := db.ivfHnswIndex != nil && db.useIVFHNSWIndex
	db.mu.RUnlock()

	// 1. 数据规模和索引可用性综合判断
	if vectorCount < 1000 {
		return StrategyBruteForce
	}

	// 2. 优先考虑IVF-HNSW混合索引（最优性能）
	if ivfHnswIndexReady && vectorCount > 50000 {
		// IVF-HNSW适合大规模、高维数据，平衡精度和速度
		if vectorDim >= 256 && ctx.QualityLevel > 0.75 {
			logger.Trace("选择IVF-HNSW混合策略：数据量=%d，质量要求=%.2f，维度=%d", vectorCount, ctx.QualityLevel, vectorDim)
			return StrategyIVFHNSW
		}
	}

	// 3. 优先考虑增强型索引（根据数据特征和性能要求）
	if vectorCount > 10000 {
		// 大规模数据集优先选择
		if ivfIndexReady && db.ivfConfig != nil {
			// IVF适合精确搜索和中高维数据
			if ctx.QualityLevel > 0.8 || (vectorDim >= 128 && vectorDim <= 2048) {
				logger.Trace("选择EnhancedIVF策略：数据量=%d，质量要求=%.2f，维度=%d", vectorCount, ctx.QualityLevel, vectorDim)
				return StrategyEnhancedIVF
			}
		}

		if lshIndexReady && db.LshConfig != nil {
			// LSH适合高维数据和快速近似搜索
			if vectorDim > 512 || (ctx.QualityLevel < 0.85 && ctx.Timeout > 0 && ctx.Timeout < 50*time.Millisecond) {
				logger.Trace("选择EnhancedLSH策略：数据量=%d，维度=%d，质量要求=%.2f", vectorCount, vectorDim, ctx.QualityLevel)
				return StrategyEnhancedLSH
			}
		}
	}

	// 4. GPU加速判断 - 新增GPU优先策略
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	if db.HardwareCaps.HasGPU && gpuAccelerator != nil && vectorCount > 10000 {
		// 对于大规模数据，优先考虑IVF-HNSW混合索引
		if vectorCount > 100000 && ivfHnswIndexReady {
			return StrategyIVFHNSW
		}
		// 其次考虑GPU混合策略
		if vectorCount > 100000 {
			return StrategyHybrid
		}
		// 中等规模数据，根据质量要求选择
		if ctx.QualityLevel > 0.8 {
			return StrategyHybrid
		}
	}

	// 5. 维度特征判断
	if vectorDim > 2048 {
		// 超高维数据优先考虑IVF-HNSW混合索引
		if ivfHnswIndexReady {
			logger.Trace("超高维数据选择IVF-HNSW混合策略：维度=%d", vectorDim)
			return StrategyIVFHNSW
		}
		// 其次考虑LSH
		if lshIndexReady && db.LshConfig != nil {
			logger.Trace("超高维数据选择EnhancedLSH策略：维度=%d", vectorDim)
			return StrategyEnhancedLSH
		}
		// 再次考虑PQ压缩
		if db.usePQCompression && db.pqCodebook != nil {
			return StrategyPQ
		}
		// 最后考虑HNSW
		if db.useHNSWIndex && db.indexed && db.hnsw != nil {
			return StrategyHNSW
		}
	}

	// 6. 质量要求判断
	if ctx.QualityLevel > 0.9 {
		// 高质量要求，优先精确搜索
		if ivfHnswIndexReady && vectorCount > 10000 {
			logger.Trace("高质量要求选择IVF-HNSW混合策略：质量要求=%.2f", ctx.QualityLevel)
			return StrategyIVFHNSW
		}
		if ivfIndexReady && db.ivfConfig != nil {
			logger.Trace("高质量要求选择EnhancedIVF策略：质量要求=%.2f", ctx.QualityLevel)
			return StrategyEnhancedIVF
		}
		if vectorCount < 100000 {
			if db.useHNSWIndex && db.indexed && db.hnsw != nil {
				return StrategyHNSW
			}
			return StrategyBruteForce
		}
		if db.useHNSWIndex && db.indexed && db.hnsw != nil {
			return StrategyHNSW
		}
	}

	// 7. 性能要求判断 (低延迟)
	if ctx.Timeout > 0 && ctx.Timeout < 10*time.Millisecond {
		// 低延迟要求，优先快速策略
		if lshIndexReady && db.LshConfig != nil {
			logger.Trace("低延迟要求选择EnhancedLSH策略：超时限制=%v", ctx.Timeout)
			return StrategyEnhancedLSH
		}
		if db.usePQCompression && db.pqCodebook != nil {
			return StrategyPQ
		}
	}

	// 8. 中等延迟要求判断
	if ctx.Timeout > 0 && ctx.Timeout < 100*time.Millisecond {
		if ivfHnswIndexReady && ctx.QualityLevel > 0.8 {
			logger.Trace("中等延迟要求选择IVF-HNSW混合策略：超时限制=%v，质量要求=%.2f", ctx.Timeout, ctx.QualityLevel)
			return StrategyIVFHNSW
		}
		if ivfIndexReady && db.ivfConfig != nil && ctx.QualityLevel > 0.7 {
			logger.Trace("中等延迟要求选择EnhancedIVF策略：超时限制=%v，质量要求=%.2f", ctx.Timeout, ctx.QualityLevel)
			return StrategyEnhancedIVF
		}
	}

	// 9. 硬件能力判断
	if db.HardwareCaps.HasGPU && vectorCount > 50000 {
		if ivfHnswIndexReady {
			return StrategyIVFHNSW // IVF-HNSW混合索引
		}
		return StrategyHybrid // GPU加速的混合策略
	}

	// 10. 默认策略选择（按优先级）
	// 优先选择IVF-HNSW混合索引
	if ivfHnswIndexReady {
		logger.Trace("默认选择IVF-HNSW混合策略")
		return StrategyIVFHNSW
	}

	// 其次选择增强型索引
	if ivfIndexReady && db.ivfConfig != nil {
		logger.Trace("默认选择EnhancedIVF策略")
		return StrategyEnhancedIVF
	}

	if lshIndexReady && db.LshConfig != nil {
		logger.Trace("默认选择EnhancedLSH策略")
		return StrategyEnhancedLSH
	}

	// 传统索引作为备选
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		return StrategyHNSW
	}

	// 检查普通的IVF索引
	if db.indexed && len(db.clusters) > 0 {
		return StrategyIVF
	}

	return StrategyBruteForce
}

// OptimizedSearch 优化的搜索方法
func (db *VectorDB) OptimizedSearch(query []float64, k int, options entity.SearchOptions) ([]entity.Result, error) {
	// 查询向量维度检查
	if len(query) != db.vectorDim {
		return nil, fmt.Errorf("查询向量维度 %d 与数据库向量维度 %d 不匹配", len(query), db.vectorDim)
	}

	// 性能监控
	var tracker *OperationTracker
	if db.performanceMonitor != nil {
		tracker = db.performanceMonitor.StartOperation("vector_search")
		tracker.AddTag("k", fmt.Sprintf("%d", k))
		tracker.AddTag("vector_dim", fmt.Sprintf("%d", len(query)))
		defer func() {
			if tracker != nil {
				tracker.End(true, nil)
			}
		}()
	}

	// 数据预处理
	processedQuery := query
	if options.NormalizeVectors && db.dataPreprocessor != nil {
		processedQuery = db.dataPreprocessor.Normalize([][]float64{query})[0]
	}

	// 硬件加速检查
	var accelerator acceler.UnifiedAccelerator
	if db.hardwareManager != nil {
		// 创建工作负载配置文件
		workloadType := acceler.WorkloadBalanced

		// 根据搜索选项确定工作负载类型
		if options.SearchTimeout > 0 && options.SearchTimeout < 10*time.Millisecond {
			workloadType = acceler.WorkloadLowLatency
		} else if options.BatchSize > 100 {
			workloadType = acceler.WorkloadHighThroughput
		} else if options.DistributedSearch {
			workloadType = acceler.WorkloadDistributed
		} else if options.PersistentStorage {
			workloadType = acceler.WorkloadPersistent
		} else if options.MemoryOptimized {
			workloadType = acceler.WorkloadMemoryOptimal
		}

		// 创建工作负载配置
		workload := acceler.CreateWorkloadProfile(workloadType, options.BatchSize, db.vectorDim)
		workload.BatchSize = len(db.vectors)

		// 显式指定加速器类型
		if options.UseGPU {
			if gpu, exists := db.hardwareManager.GetAccelerator(acceler.AcceleratorGPU); exists && gpu.IsAvailable() {
				accelerator = gpu
				logger.Debug("根据用户指定使用GPU加速器")
				if tracker != nil {
					tracker.AddTag("accelerator", "gpu_explicit")
				}
			}
		} else if options.UseFPGA {
			if fpga, exists := db.hardwareManager.GetAccelerator(acceler.AcceleratorFPGA); exists && fpga.IsAvailable() {
				accelerator = fpga
				logger.Debug("根据用户指定使用FPGA加速器")
				if tracker != nil {
					tracker.AddTag("accelerator", "fpga_explicit")
				}
			}
		} else {
			// 自动选择最佳加速器
			bestAcc := db.hardwareManager.GetOptimalAccelerator(workload)
			if bestAcc != nil && bestAcc.IsAvailable() {
				accelerator = bestAcc
				logger.Debug("自动选择最佳加速器: %s", bestAcc.GetType())
				if tracker != nil {
					tracker.AddTag("accelerator", "auto_selected_"+bestAcc.GetType())
					tracker.AddTag("workload_type", workloadType)
				}
			}
		}
	}

	// 缓存检查
	var cacheKey string
	if options.UseCache && db.cacheManager != nil {
		cacheKey = db.generateCacheKey(processedQuery, k, options)
		if strategy, exists := db.cacheManager.GetStrategy(options.CacheStrategy); exists {
			if cached, found := strategy.Get(cacheKey); found {
				if results, ok := cached.([]entity.Result); ok {
					if tracker != nil {
						tracker.AddTag("cache_hit", "true")
					}
					return results, nil
				}
			}
		}
	}

	// 硬件加速优先
	if accelerator != nil {
		db.mu.RLock()
		// 1. 构建加速器需要的数据集和ID映射
		database := make([][]float64, 0, len(db.vectors))
		idMap := make(map[int]string, len(db.vectors))
		i := 0
		for id, vec := range db.vectors {
			database = append(database, vec)
			idMap[i] = id
			i++
		}
		db.mu.RUnlock()

		// 2. 调用硬件加速搜索
		accelOpts := entity.SearchOptions{K: k}
		// 传递更多选项给加速器
		accelOpts.BatchSize = options.BatchSize
		accelOpts.QualityLevel = options.QualityLevel
		accelOpts.SearchTimeout = options.SearchTimeout

		if acceleratedResults, accelErr := accelerator.AccelerateSearch(processedQuery, database, accelOpts); accelErr == nil {
			// 3. 处理加速结果
			results := make([]entity.Result, len(acceleratedResults))
			for i, res := range acceleratedResults {
				results[i] = entity.Result{
					Id:       idMap[res.Index],
					Distance: res.Distance,
				}
			}

			// 缓存结果
			if options.UseCache && db.cacheManager != nil && cacheKey != "" {
				if strategy, exists := db.cacheManager.GetStrategy(options.CacheStrategy); exists {
					strategy.Put(cacheKey, results)
				}
			}

			if tracker != nil {
				tracker.AddTag("hardware_accelerated", "true")
				tracker.AddTag("accelerator_type", accelerator.GetType())
				tracker.End(true, nil)
			}
			return results, nil
		} else {
			logger.Warning("硬件加速搜索失败，回退到软件搜索: %v", accelErr)
			if tracker != nil {
				tracker.AddTag("hardware_fallback", "true")
				tracker.AddTag("hardware_error", accelErr.Error())
			}
		}
	}

	// 构建搜索上下文
	ctx := SearchContext{
		QueryVector:  processedQuery,
		K:            k,
		Nprobe:       options.Nprobe,
		Timeout:      options.SearchTimeout,
		QualityLevel: options.QualityLevel,
	}

	// 如果没有设置质量等级，使用默认值
	if ctx.QualityLevel == 0 {
		ctx.QualityLevel = 0.8
	}

	// 策略选择：优先使用强制策略，否则智能选择
	var indexStrategy IndexStrategy
	if options.ForceStrategy != "" {
		// 使用强制指定的策略
		switch options.ForceStrategy {
		case "brute_force":
			indexStrategy = StrategyBruteForce
		case "ivf":
			indexStrategy = StrategyIVF
		case "hnsw":
			indexStrategy = StrategyHNSW
		case "pq":
			indexStrategy = StrategyPQ
		case "lsh":
			indexStrategy = StrategyEnhancedLSH
		case "ivf_hnsw":
			indexStrategy = StrategyIVFHNSW
		case "hybrid":
			indexStrategy = StrategyHybrid
		default:
			indexStrategy = db.selectOptimalStrategyWithAdaptive(ctx)
		}
		if tracker != nil {
			tracker.AddTag("forced_strategy", options.ForceStrategy)
		}
	} else {
		// 智能策略选择
		indexStrategy = db.selectOptimalStrategyWithAdaptive(ctx)
	}

	logger.Trace("选择搜索策略: %v, 数据量: %d, 维度: %d, 质量要求: %.2f",
		indexStrategy, len(db.vectors), len(query), ctx.QualityLevel)

	startTime := time.Now()
	var results []entity.Result
	var err error

	switch indexStrategy {
	case StrategyBruteForce:
		results, err = db.bruteForceSearch(processedQuery, k)
	case StrategyIVF:
		results, err = db.ivfSearchWithScores(processedQuery, k, ctx.Nprobe, db.GetOptimalStrategy(processedQuery))
	case StrategyHNSW:
		results, err = db.hnswSearchWithScores(processedQuery, k)
	case StrategyPQ:
		results, err = db.pqSearchWithScores(processedQuery, k)
	case StrategyHybrid:
		results, err = db.hybridSearchWithScores(processedQuery, k, ctx)
	case StrategyEnhancedIVF:
		results, err = db.EnhancedIVFSearch(processedQuery, k, ctx.Nprobe)
		if err != nil {
			logger.Warning("EnhancedIVF搜索失败，回退到传统IVF: %v", err)
			results, err = db.ivfSearchWithScores(processedQuery, k, ctx.Nprobe, db.GetOptimalStrategy(processedQuery))
		}
	case StrategyEnhancedLSH:
		results, err = db.EnhancedLSHSearch(processedQuery, k)
		if err != nil {
			logger.Warning("EnhancedLSH搜索失败，回退到传统搜索: %v", err)
			// 根据数据规模选择回退策略
			if len(db.vectors) > 10000 {
				results, err = db.ivfSearchWithScores(processedQuery, k, ctx.Nprobe, db.GetOptimalStrategy(processedQuery))
			} else {
				results, err = db.bruteForceSearch(processedQuery, k)
			}
		}
	case StrategyIVFHNSW:
		results, err = db.ivfHnswSearchWithScores(processedQuery, k, ctx)
		if err != nil {
			logger.Warning("IVF-HNSW混合搜索失败，回退到增强IVF: %v", err)
			// 回退到增强IVF或传统搜索
			if db.ivfIndex != nil && db.ivfIndex.Enable {
				results, err = db.EnhancedIVFSearch(processedQuery, k, ctx.Nprobe)
			} else if len(db.vectors) > 10000 {
				results, err = db.ivfSearchWithScores(processedQuery, k, ctx.Nprobe, db.GetOptimalStrategy(processedQuery))
			} else {
				results, err = db.bruteForceSearch(processedQuery, k)
			}
		}
	default:
		results, err = db.ivfSearchWithScores(processedQuery, k, ctx.Nprobe, db.GetOptimalStrategy(processedQuery))
	}

	// 多阶段搜索优化
	if options.EnableMultiStage && err == nil && len(results) > 0 {
		if options.CoarseK > k && options.CoarseK <= len(results)*2 {
			// 第一阶段：粗搜索获取更多候选
			coarseResults, coarseErr := db.executeSearch(processedQuery, options.CoarseK, indexStrategy, ctx)
			if coarseErr == nil && len(coarseResults) > k {
				// 第二阶段：精搜索从候选中选择最佳结果
				if options.EnablePQRefinement && db.pqCodebook != nil {
					// 使用PQ解码进行精确重排
					refinedResults := db.refinePQResults(processedQuery, coarseResults[:min(options.CoarseK, len(coarseResults))], k)
					if len(refinedResults) > 0 {
						results = refinedResults
					}
				} else {
					// 使用精确距离计算重排
					results = db.reRankResults(processedQuery, coarseResults, k)
				}
				if tracker != nil {
					tracker.AddTag("multi_stage", "true")
				}
			}
		}
	}

	// 结果后处理
	if err == nil && len(results) > 0 {
		// 应用过滤器
		if options.FilterFirst && db.dataPreprocessor != nil {
			results = db.dataPreprocessor.FilterResults(results, options)
		}

		// 结果去重
		if options.EnableDeduplication {
			results = db.deduplicateResults(results)
		}

		// 缓存结果
		if options.UseCache && db.cacheManager != nil && cacheKey != "" {
			if strategy, exists := db.cacheManager.GetStrategy(options.CacheStrategy); exists {
				strategy.Put(cacheKey, results)
				if tracker != nil {
					tracker.AddTag("cache_stored", "true")
				}
			}
		}
	}

	// 记录增强的性能指标
	latency := time.Since(startTime)
	quality := db.estimateSearchQuality(results, ctx)
	db.updateEnhancedPerformanceMetrics(indexStrategy, latency, len(results), quality, ctx)

	// 性能监控记录
	if tracker != nil {
		tracker.AddTag("strategy", fmt.Sprintf("%v", indexStrategy))
		tracker.AddTag("result_count", fmt.Sprintf("%d", len(results)))
		tracker.AddTag("latency_ms", fmt.Sprintf("%.2f", float64(latency.Nanoseconds())/1e6))
		tracker.AddTag("quality", fmt.Sprintf("%.3f", quality))
		var errorTags map[string]string
		if err != nil {
			errorTags = map[string]string{"error": err.Error()}
		}
		tracker.End(err == nil, errorTags)
	}

	return results, err
}

// MonitorGPUHealth GPU健康状态监控
func (db *VectorDB) MonitorGPUHealth() {
	if db.hardwareManager == nil {
		return
	}
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	if !db.HardwareCaps.HasGPU || gpuAccelerator == nil {
		return
	}

	logger.Info("启动GPU健康监控服务")

	// 健康检查间隔
	healthCheckTicker := time.NewTicker(30 * time.Second) // 每30秒检查一次
	// 性能报告间隔
	performanceReportTicker := time.NewTicker(5 * time.Minute) // 每5分钟报告一次性能

	defer func() {
		healthCheckTicker.Stop()
		performanceReportTicker.Stop()
	}()

	// 连续失败计数器
	failureCount := 0
	const maxFailures = 3 // 连续失败3次后禁用GPU

	for {
		select {
		case <-healthCheckTicker.C:
			// 执行GPU健康检查
			if err := db.CheckGPUStatus(); err != nil {
				failureCount++
				logger.Warning("GPU健康检查失败(%d/%d): %v", failureCount, maxFailures, err)

				// 更新统计信息
				db.statsMu.Lock()
				db.stats.GPUErrors++
				db.statsMu.Unlock()

				// 连续失败达到阈值，禁用GPU加速
				if failureCount >= maxFailures {
					logger.Error("GPU连续%d次健康检查失败，自动禁用GPU加速", maxFailures)
					db.HardwareCaps.HasGPU = false

					// 尝试重新初始化GPU加速器
					go func() {
						time.Sleep(5 * time.Minute) // 等待5分钟后尝试恢复
						logger.Info("尝试恢复GPU加速...")
						if db.hardwareManager != nil {
							config := db.hardwareManager.GetConfig()
							if config != nil && config.GPU.Enable {
								if err := db.InitializeGPUAccelerator(config.GPU.DeviceID, "Flat"); err == nil {
									logger.Info("GPU加速已恢复")
									db.HardwareCaps.HasGPU = true
									failureCount = 0
								} else {
									logger.Warning("GPU加速恢复失败: %v", err)
								}
							}
						}
					}()
				}
			} else {
				// 健康检查成功，重置失败计数
				if failureCount > 0 {
					logger.Info("GPU健康状态已恢复正常")
					failureCount = 0
				}
			}

		case <-performanceReportTicker.C:
			// 定期报告GPU性能指标
			db.reportGPUPerformanceMetrics()

		case <-db.stopCh:
			logger.Info("停止GPU健康监控服务")
			return
		}
	}
}

// reportGPUPerformanceMetrics 报告GPU性能指标
func (db *VectorDB) reportGPUPerformanceMetrics() {
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	if !db.HardwareCaps.HasGPU || gpuAccelerator == nil {
		return
	}

	// 获取GPU统计信息
	db.statsMu.RLock()
	gpuStats := db.stats
	db.statsMu.RUnlock()

	// 计算GPU利用率
	gpuUtilization := 0.0
	if gpuStats.TotalQueries > 0 {
		gpuUtilization = float64(gpuStats.GPUComputations+gpuStats.GPUSearches) / float64(gpuStats.TotalQueries) * 100
	}

	// 计算GPU错误率
	gpuErrorRate := 0.0
	totalGPUOperations := gpuStats.GPUComputations + gpuStats.GPUSearches
	if totalGPUOperations > 0 {
		gpuErrorRate = float64(gpuStats.GPUErrors) / float64(totalGPUOperations) * 100
	}

	// 计算平均GPU时间
	avgGPUTime := time.Duration(0)
	if gpuStats.GPUComputations > 0 {
		avgGPUTime = time.Duration(gpuStats.TotalGPUTime / gpuStats.GPUComputations)
	}

	// 计算平均GPU搜索时间
	avgGPUSearchTime := time.Duration(0)
	if gpuStats.GPUSearches > 0 {
		avgGPUSearchTime = time.Duration(gpuStats.TotalGPUSearchTime / gpuStats.GPUSearches)
	}

	// 获取GPU内存信息
	memoryInfo := "未知"
	if gpuAccel, ok := gpuAccelerator.(*acceler.FAISSAccelerator); ok {
		free, total, err := gpuAccel.GetGPUMemoryInfo()
		if err == nil {
			memUsagePercent := float64(total-free) / float64(total) * 100
			memoryInfo = fmt.Sprintf("%.2f%% (可用: %d MB / 总计: %d MB)",
				memUsagePercent, free/(1024*1024), total/(1024*1024))
		}
	}

	// 报告性能指标
	logger.Info("===== GPU性能报告 =====")
	logger.Info("GPU计算次数: %d", gpuStats.GPUComputations)
	logger.Info("GPU搜索次数: %d", gpuStats.GPUSearches)
	logger.Info("GPU错误次数: %d", gpuStats.GPUErrors)
	logger.Info("GPU利用率: %.2f%%", gpuUtilization)
	logger.Info("GPU错误率: %.2f%%", gpuErrorRate)
	logger.Info("平均GPU计算时间: %v", avgGPUTime)
	logger.Info("平均GPU搜索时间: %v", avgGPUSearchTime)
	logger.Info("GPU内存使用: %s", memoryInfo)
	logger.Info("GPU加速状态: %s", db.GetGPUAccelerationStatus())
	logger.Info("=========================")
}

// InitializeGPUAccelerator 初始化GPU加速器
func (db *VectorDB) InitializeGPUAccelerator(deviceID int, indexType string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if !db.HardwareCaps.HasGPU {
		return fmt.Errorf("系统不支持GPU加速")
	}

	// 检查硬件管理器是否初始化
	if db.hardwareManager == nil {
		return fmt.Errorf("硬件管理器未初始化")
	}

	// 如果已经初始化过，先关闭旧的加速器
	if existingAccel := db.hardwareManager.GetGPUAccelerator(); existingAccel != nil {
		logger.Info("关闭现有GPU加速器")
		// 这里可以添加关闭逻辑，如果加速器支持的话
		if closer, ok := existingAccel.(interface{ Cleanup() error }); ok {
			if err := closer.Cleanup(); err != nil {
				logger.Warning("关闭现有GPU加速器失败: %v", err)
			}
		}
	}

	// 记录初始化开始时间
	startTime := time.Now()

	// 创建FAISS GPU加速器
	var gpuAccel acceler.Accelerator

	// 优先使用硬件管理器获取GPU加速器
	if db.hardwareManager != nil {
		if acc, exists := db.hardwareManager.GetAccelerator(acceler.AcceleratorGPU); exists && acc.IsAvailable() {
			gpuAccel = acc
			logger.Info("使用硬件管理器获取GPU加速器")
		}
	}

	// 如果硬件管理器未提供GPU加速器，则直接创建
	if gpuAccel == nil {
		gpuAccel = acceler.NewFAISSAccelerator(deviceID, indexType)
		logger.Info("直接创建FAISS GPU加速器")
	}

	if err := gpuAccel.Initialize(); err != nil {
		// 更新统计信息
		db.statsMu.Lock()
		db.stats.GPUErrors++
		db.statsMu.Unlock()

		return fmt.Errorf("GPU加速器初始化失败: %w", err)
	}

	// 计算初始化耗时
	initDuration := time.Since(startTime)

	// 更新统计信息
	db.statsMu.Lock()
	db.stats.TotalGPUTime += initDuration.Milliseconds()
	db.stats.GPUComputations++
	db.statsMu.Unlock()

	// 将GPU加速器注册到硬件管理器
	if err := db.hardwareManager.RegisterGPUAccelerator(gpuAccel); err != nil {
		return fmt.Errorf("注册GPU加速器到硬件管理器失败: %w", err)
	}
	logger.Info("GPU加速器初始化成功，耗时: %v", initDuration)

	// 启动GPU健康监控（如果尚未启动）
	if db.stopCh == nil {
		db.stopCh = make(chan struct{})
		go db.MonitorGPUHealth()
	}

	return nil
}

// CheckGPUStatus 检查GPU状态和可用性
func (db *VectorDB) CheckGPUStatus() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.HardwareCaps.HasGPU {
		return fmt.Errorf("系统未检测到GPU支持")
	}

	if db.hardwareManager == nil {
		return fmt.Errorf("硬件管理器未初始化")
	}

	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	if gpuAccelerator == nil {
		return fmt.Errorf("GPU加速器未初始化")
	}

	// 如果是FAISS GPU加速器，进行详细检查
	if gpuAccel, ok := gpuAccelerator.(*acceler.FAISSAccelerator); ok {
		// 检查GPU可用性
		if err := gpuAccel.CheckGPUAvailability(); err != nil {
			// 记录详细错误信息
			logger.Error("GPU可用性检查失败: %v", err)

			// 获取GPU内存信息用于诊断
			free, total, memErr := gpuAccel.GetGPUMemoryInfo()
			if memErr == nil {
				logger.Info("GPU内存状态: 可用 %d MB / 总计 %d MB (%.2f%%)",
					free/(1024*1024),
					total/(1024*1024),
					float64(free)/float64(total)*100)
			}

			// 更新统计信息
			db.statsMu.Lock()
			db.stats.GPUErrors++
			db.statsMu.Unlock()

			return fmt.Errorf("GPU可用性检查失败: %w", err)
		}

		// 记录GPU内存状态
		free, total, memErr := gpuAccel.GetGPUMemoryInfo()
		if memErr != nil {
			// 更新统计信息
			db.statsMu.Lock()
			db.stats.GPUErrors++
			db.statsMu.Unlock()

			return fmt.Errorf("获取GPU内存信息失败: %w", memErr)
		}

		// 计算内存使用率
		memUsagePercent := float64(total-free) / float64(total) * 100

		// 更新统计信息
		db.statsMu.Lock()
		db.stats.GPUUtilization = memUsagePercent
		db.statsMu.Unlock()

		// 记录内存状态
		logger.Debug("GPU内存状态正常: 可用 %d MB / 总计 %d MB (使用率: %.2f%%)",
			free/(1024*1024),
			total/(1024*1024),
			memUsagePercent)

		// 检查内存使用率是否过高
		if memUsagePercent > 95 {
			logger.Warning("GPU内存使用率过高 (%.2f%%)，可能影响性能", memUsagePercent)
		}

		return nil
	} else {
		// 非FAISS加速器，只进行基本检查
		logger.Debug("使用非FAISS GPU加速器，跳过详细检查")
		return nil
	}
}

// pqSearchWithScores PQ压缩搜索
func (db *VectorDB) pqSearchWithScores(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.usePQCompression || db.pqCodebook == nil {
		return nil, fmt.Errorf("PQ压缩未启用")
	}

	results := make([]entity.Result, 0, len(db.compressedVectors))
	// 获取最优计算策略
	selectStrategy := db.GetOptimalStrategy(query)

	// 并行计算PQ近似距离
	numWorkers := runtime.NumCPU()
	workChan := make(chan string, len(db.compressedVectors))
	resultChan := make(chan entity.Result, len(db.compressedVectors))
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for id := range workChan {
				if compVec, exists := db.compressedVectors[id]; exists {
					// 使用自适应距离计算优化PQ搜索
					dist, err := db.CalculateApproximateDistancePQWithStrategy(query, compVec, selectStrategy)
					if err == nil {
						similarity := 1.0 / (1.0 + dist) // 转换为相似度
						resultChan <- entity.Result{
							Id:         id,
							Similarity: similarity,
						}
					}
				}
			}
		}()
	}

	// 发送任务
	for id := range db.compressedVectors {
		workChan <- id
	}
	close(workChan)

	wg.Wait()
	close(resultChan)

	// 收集结果
	for result := range resultChan {
		results = append(results, result)
	}

	// 排序并返回top-k
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if k > len(results) {
		k = len(results)
	}

	return results[:k], nil
}

// CalculateApproximateDistancePQWithStrategy 使用指定策略计算PQ近似距离
func (db *VectorDB) CalculateApproximateDistancePQWithStrategy(query []float64, compressedVector entity.CompressedVector, strategy acceler.ComputeStrategy) (float64, error) {
	if compressedVector.Data == nil || len(compressedVector.Data) == 0 {
		return 0, fmt.Errorf("压缩向量数据不能为空")
	}
	if len(compressedVector.Data) != db.numSubVectors {
		return 0, fmt.Errorf("压缩向量的数据长度 %d 与子向量数量 %d 不匹配", len(compressedVector.Data), db.numSubVectors)
	}

	subVectorDim := len(query) / db.numSubVectors
	totalSquaredDistance := 0.0

	// 对每个子向量使用自适应查找最近质心
	for m := 0; m < db.numSubVectors; m++ {
		// 获取查询向量的子向量
		start := m * subVectorDim
		end := start + subVectorDim
		querySubVector := query[start:end]

		// 获取压缩向量中对应的质心索引
		centroidIndex := int(compressedVector.Data[m])

		if centroidIndex < 0 || centroidIndex >= len(db.pqCodebook[m]) {
			return 0, fmt.Errorf("子空间 %d 的质心索引 %d 超出范围", m, centroidIndex)
		}

		// 获取对应的质心
		centroid := db.pqCodebook[m][centroidIndex]

		// 使用自适应策略计算距离
		_, dist := acceler.AdaptiveFindNearestCentroid(querySubVector, []entity.Point{centroid}, strategy)
		totalSquaredDistance += dist
	}

	return totalSquaredDistance, nil
}

// hnswSearchWithScores HNSW搜索并返回带分数的结果
func (db *VectorDB) hnswSearchWithScores(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.useHNSWIndex || !db.indexed || db.hnsw == nil {
		return nil, fmt.Errorf("HNSW索引未启用或未构建")
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("数据库为空，无法进行搜索")
	}

	// 选择最优计算策略
	selectStrategy := db.GetOptimalStrategy(query)

	// 根据策略设置距离函数
	switch selectStrategy {
	case acceler.StrategyAVX512, acceler.StrategyAVX2:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := acceler.AdaptiveCosineSimilarity(a, b, selectStrategy)
			return 1.0 - sim, nil
		})
	default:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := acceler.CosineSimilarity(a, b)
			return 1.0 - sim, nil
		})
	}

	// 执行搜索
	normalizedQuery := acceler.NormalizeVector(query)
	hnswResults, err := db.hnsw.Search(normalizedQuery, k)
	if err != nil {
		return nil, fmt.Errorf("HNSW搜索失败: %v", err)
	}

	// 转换为entity.Result格式
	results := make([]entity.Result, 0, len(hnswResults))
	for _, hnswResult := range hnswResults {
		// HNSW返回的是距离，需要转换为相似度
		similarity := 1.0 - hnswResult.Similarity
		if similarity < 0 {
			similarity = 0
		}

		results = append(results, entity.Result{
			Id:         hnswResult.Id,
			Similarity: similarity,
		})
	}

	// 按相似度降序排序
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// hybridSearchWithScores 混合搜索策略
func (db *VectorDB) hybridSearchWithScores(query []float64, k int, ctx SearchContext) ([]entity.Result, error) {
	// 第一阶段：使用PQ进行快速粗排
	coarseK := k * 10 // 粗排返回更多候选
	if coarseK > len(db.vectors) {
		coarseK = len(db.vectors)
	}

	var candidates []entity.Result
	var err error

	if db.usePQCompression && db.pqCodebook != nil {
		candidates, err = db.pqSearchWithScores(query, coarseK)
		if err != nil {
			logger.Warning("PQ粗排失败，回退到IVF: %v", err)
			candidates, err = db.ivfSearchWithScores(query, coarseK, ctx.Nprobe, db.GetOptimalStrategy(query))
		}
	} else {
		candidates, err = db.ivfSearchWithScores(query, coarseK, ctx.Nprobe, db.GetOptimalStrategy(query))
	}

	if err != nil {
		return nil, err
	}

	// 第二阶段：精确重排
	candidateIDs := make([]string, len(candidates))
	for i, candidate := range candidates {
		candidateIDs[i] = candidate.Id
	}

	// 检查是否可以使用GPU加速精排
	if db.shouldUseGPUBatchSearch(1, len(candidateIDs)) && len(candidateIDs) > 100 {
		return db.gpuFineRanking(query, candidateIDs, k)
	}

	// 使用CPU SIMD加速精排
	optimalStrategy := db.strategyComputeSelector.SelectOptimalStrategy(len(candidateIDs), len(query))
	finalResults, err := db.fineRankingWithScores(query, candidateIDs, k, optimalStrategy)
	if err != nil {
		return nil, err
	}

	return finalResults, nil
}

// updatePerformanceMetrics 更新性能指标
func (db *VectorDB) updatePerformanceMetrics(strategy IndexStrategy, latency time.Duration, resultCount int) {
	db.statsMu.Lock()
	defer db.statsMu.Unlock()

	// 更新总查询数
	db.stats.TotalQueries++

	// 更新平均查询时间（使用指数移动平均）
	if db.stats.AvgQueryTime == 0 {
		db.stats.AvgQueryTime = latency
	} else {
		// 使用0.1的平滑因子进行指数移动平均
		alpha := 0.1
		db.stats.AvgQueryTime = time.Duration(float64(db.stats.AvgQueryTime)*(1-alpha) + float64(latency)*alpha)
	}

	// 更新内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	db.stats.MemoryUsage = m.Alloc

	// 如果StrategySelector存在，更新策略特定的性能指标
	if db.strategySelector != nil {
		if db.strategySelector.performance == nil {
			db.strategySelector.performance = make(map[IndexStrategy]*PerformanceMetrics)
		}

		// 获取或创建策略的性能指标
		metrics, exists := db.strategySelector.performance[strategy]
		if !exists {
			metrics = &PerformanceMetrics{
				AvgLatency:    latency,
				Recall:        1.0, // 初始假设100%召回率
				ThroughputQPS: 1.0 / latency.Seconds(),
				MemoryUsage:   m.Alloc,
				LastUpdated:   time.Now(),
			}
			db.strategySelector.performance[strategy] = metrics
		} else {
			// 更新现有指标（使用指数移动平均）
			alpha := 0.1
			metrics.AvgLatency = time.Duration(float64(metrics.AvgLatency)*(1-alpha) + float64(latency)*alpha)

			// 计算当前QPS
			currentQPS := 1.0 / latency.Seconds()
			metrics.ThroughputQPS = metrics.ThroughputQPS*(1-alpha) + currentQPS*alpha

			// 更新内存使用
			metrics.MemoryUsage = m.Alloc
			metrics.LastUpdated = time.Now()

			// 根据结果数量估算召回率（简化版本）
			if resultCount > 0 {
				// 这里可以根据实际业务逻辑调整召回率计算
				estimatedRecall := math.Min(1.0, float64(resultCount)/float64(len(db.vectors)))
				metrics.Recall = metrics.Recall*(1-alpha) + estimatedRecall*alpha
			}
		}
	}

	// 记录详细日志（可选，用于调试）
	logger.Trace("性能指标更新 - 策略: %v, 延迟: %v, 结果数: %d, 总查询数: %d, 平均延迟: %v, 内存使用: %d bytes",
		strategy, latency, resultCount, db.stats.TotalQueries, db.stats.AvgQueryTime, db.stats.MemoryUsage)

	// 定期输出性能摘要（每1000次查询）
	if db.stats.TotalQueries%1000 == 0 {
		db.logPerformanceSummary()
	}
}

// logPerformanceSummary 输出性能摘要
func (db *VectorDB) logPerformanceSummary() {
	logger.Info("=== 性能摘要 (查询数: %d) ===", db.stats.TotalQueries)
	logger.Info("平均查询时间: %v", db.stats.AvgQueryTime)
	logger.Info("内存使用: %.2f MB", float64(db.stats.MemoryUsage)/1024/1024)
	logger.Info("缓存命中率: %.2f%%", float64(db.stats.CacheHits)*100/float64(db.stats.TotalQueries))

	// 输出各策略的性能表现
	if db.strategySelector != nil && db.strategySelector.performance != nil {
		logger.Info("各策略性能表现:")
		for indexStrategy, metrics := range db.strategySelector.performance {
			strategyName := db.getStrategyName(indexStrategy)
			logger.Info("  %s: 延迟=%v, QPS=%.2f, 召回率=%.2f%%, 内存=%d bytes",
				strategyName, metrics.AvgLatency, metrics.ThroughputQPS,
				metrics.Recall*100, metrics.MemoryUsage)
		}
	}
}

// getStrategyName 获取策略名称
func (db *VectorDB) getStrategyName(strategy IndexStrategy) string {
	switch strategy {
	case StrategyBruteForce:
		return "暴力搜索"
	case StrategyIVF:
		return "IVF索引"
	case StrategyHNSW:
		return "HNSW索引"
	case StrategyPQ:
		return "PQ压缩"
	case StrategyHybrid:
		return "混合策略"
	case StrategyEnhancedLSH:
		return "增强LSH"
	case StrategyEnhancedIVF:
		return "增强IVF"
	case StrategyIVFHNSW:
		return "IVF-HNSW混合"
	default:
		return "未知策略"
	}
}

// GetStrategyPerformance 获取特定策略的性能指标
func (db *VectorDB) GetStrategyPerformance(strategy IndexStrategy) *PerformanceMetrics {
	if db.strategySelector == nil || db.strategySelector.performance == nil {
		return nil
	}

	db.strategySelector.mu.RLock()
	defer db.strategySelector.mu.RUnlock()

	metrics, exists := db.strategySelector.performance[strategy]
	if !exists {
		return nil
	}

	// 返回副本以避免并发修改
	return &PerformanceMetrics{
		AvgLatency:    metrics.AvgLatency,
		Recall:        metrics.Recall,
		ThroughputQPS: metrics.ThroughputQPS,
		MemoryUsage:   metrics.MemoryUsage,
		LastUpdated:   metrics.LastUpdated,
	}
}

// GetBestPerformingStrategy 获取性能最佳的策略
func (db *VectorDB) GetBestPerformingStrategy() IndexStrategy {
	if db.strategySelector == nil || db.strategySelector.performance == nil {
		return StrategyBruteForce
	}

	db.strategySelector.mu.RLock()
	defer db.strategySelector.mu.RUnlock()

	var bestStrategy IndexStrategy
	var bestScore float64 = -1

	for indexStrategy, metrics := range db.strategySelector.performance {
		// 综合评分：考虑延迟、QPS和召回率
		// 分数 = (召回率 * QPS) / 延迟(秒)
		latencySeconds := metrics.AvgLatency.Seconds()
		if latencySeconds > 0 {
			score := (metrics.Recall * metrics.ThroughputQPS) / latencySeconds
			if score > bestScore {
				bestScore = score
				bestStrategy = indexStrategy
			}
		}
	}

	return bestStrategy
}

// AdaptiveReindex 自适应重建索引
func (db *VectorDB) AdaptiveReindex() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	vectorCount := len(db.vectors)

	// 根据数据规模选择最优索引策略
	if vectorCount > 1000000 {
		// 大规模数据，启用所有优化
		if !db.useHNSWIndex {
			db.useHNSWIndex = true
			if err := db.BuildHNSWIndex(); err != nil {
				logger.Warning("构建HNSW索引失败: %v", err)
			}
		}

		if !db.usePQCompression && db.pqCodebook != nil {
			db.usePQCompression = true
			if err := db.CompressExistingVectors(); err != nil {
				logger.Warning("PQ压缩失败: %v", err)
			}
		}
	} else if vectorCount > 100000 {
		// 中等规模数据，选择性启用优化
		if !db.useHNSWIndex {
			db.useHNSWIndex = true
			if err := db.BuildHNSWIndex(); err != nil {
				logger.Warning("构建HNSW索引失败: %v", err)
			}
		}
	}

	// 根据数据规模动态调整参数
	maxIterations := 100
	tolerance := 1e-4
	if vectorCount > 1000000 {
		maxIterations = 200 // 大数据集需要更多迭代
		tolerance = 1e-5    // 更严格的收敛条件
	} else if vectorCount < 10000 {
		maxIterations = 50 // 小数据集可以减少迭代
	}

	// 重建IVF索引
	if err := db.BuildIndex(maxIterations, tolerance); err != nil {
		return fmt.Errorf("重建IVF索引失败: %v", err)
	}

	logger.Info("自适应重建索引完成，数据量: %d", vectorCount)
	return nil
}

func (db *VectorDB) IsHNSWEnabled() bool {
	return db.useHNSWIndex
}

func (db *VectorDB) GetBackupPath() string {
	return db.backupPath
}

func (db *VectorDB) IsPQCompressionEnabled() bool {
	return db.useCompression
}

func (db *VectorDB) Close() {
	db.mu.Lock()
	defer db.mu.Unlock()

	logger.Info("Closing VectorDB...")
	// 发送停止信号给后台任务
	if db.stopCh != nil {
		close(db.stopCh) // 关闭channel以通知goroutine停止
		logger.Info("Stop signal sent to background tasks.")
	}
	// 尝试保存数据到文件
	if db.backupPath != "" {
		logger.Info("Attempting to save VectorDB data to %s before closing...", db.filePath)
		if err := db.SaveToFile(db.backupPath); err != nil {
			logger.Error("Error saving VectorDB data to %s: %v", db.filePath, err)
		}
	}

	// 清理内存中的数据结构
	db.vectors = make(map[string][]float64) // 清空向量
	db.clusters = make([]Cluster, 0)        // 清空簇信息
	db.indexed = false                      // 重置索引状态

	db.invertedMu.Lock()
	db.invertedIndex = make(map[string][]string) // 清空倒排索引
	db.invertedMu.Unlock()

	db.normalizedVectors = make(map[string][]float64)               // 清空归一化向量
	db.compressedVectors = make(map[string]entity.CompressedVector) // 清空压缩向量

	// 重置其他可能的状态字段
	db.vectorDim = 0

	logger.Info("VectorDB closed successfully.")
}

func (db *VectorDB) IsIndexed() bool {
	return db.indexed
}

// NewVectorDB 创建一个新的 VectorDB 实例。
// 如果 filePath 非空且文件存在，则尝试从中加载数据。
// numClusters 指定了用于索引的簇数量，如果 <=0，则不启用索引功能。
// NewVectorDBWithDimension 创建一个具有指定维度和距离类型的 VectorDB 实例
func NewVectorDBWithDimension(dimension int, distanceType string) (*VectorDB, error) {
	// 将字符串距离类型转换为枚举值
	var distanceTypeEnum int
	switch distanceType {
	case "euclidean":
		distanceTypeEnum = enum.EuclideanDistance
	case "cosine":
		distanceTypeEnum = enum.CosineDistance
	case "manhattan":
		distanceTypeEnum = enum.ManhattanDistance
	case "dot":
		distanceTypeEnum = enum.DotProduct
	case "euclidean_squared":
		distanceTypeEnum = enum.EuclideanDistanceSquared
	case "cosine_similarity":
		distanceTypeEnum = enum.CosineSimilarity
	default:
		return nil, fmt.Errorf("不支持的距离类型: %s", distanceType)
	}

	// 创建距离计算器
	distanceCalculator, err := algorithm.NewDistanceCalculator(distanceTypeEnum)
	if err != nil {
		return nil, err
	}

	// 创建 VectorDB 实例
	db := &VectorDB{
		vectors:            make(map[string][]float64),
		filePath:           "", // 空文件路径
		numClusters:        10, // 默认簇数
		clusters:           make([]Cluster, 0),
		indexed:            false,
		invertedIndex:      make(map[string][]string),
		vectorDim:          dimension,
		vectorizedType:     DefaultVectorized,
		normalizedVectors:  make(map[string][]float64),
		config:             AdaptiveConfig{},
		stats:              PerformanceStats{},
		distanceCalculator: distanceCalculator,

		// 初始化 PQ 相关字段
		pqCodebook:               nil,
		numSubVectors:            0,
		numCentroidsPerSubVector: 0,
		usePQCompression:         false,
		stopCh:                   make(chan struct{}),
	}

	return db, nil
}

func NewVectorDB(filePath string, numClusters int) *VectorDB {
	// 创建默认的欧几里得距离计算器
	distanceCalculator, _ := algorithm.NewDistanceCalculator(enum.EuclideanDistance)

	db := &VectorDB{
		vectors:            make(map[string][]float64),
		filePath:           filePath,
		numClusters:        numClusters,
		clusters:           make([]Cluster, 0),
		indexed:            false,
		invertedIndex:      make(map[string][]string),
		vectorDim:          0,
		vectorizedType:     DefaultVectorized,
		normalizedVectors:  make(map[string][]float64),
		config:             AdaptiveConfig{},
		stats:              PerformanceStats{},
		distanceCalculator: distanceCalculator, // 初始化距离计算器

		// 初始化 PQ 相关字段
		pqCodebook:               nil,
		numSubVectors:            0, // 默认为0，表示未配置或不使用
		numCentroidsPerSubVector: 0, // 默认为0
		usePQCompression:         false,
		stopCh:                   make(chan struct{}), // 初始化stopCh

		useHNSWIndex:   false,
		maxConnections: 16,    // 默认值
		efConstruction: 100.0, // 默认值
		efSearch:       50.0,  // 默认值

		// 初始化硬件自适应组件
		strategyComputeSelector: acceler.NewComputeStrategySelector(),
		currentStrategy:         acceler.StrategyStandard,
		strategySelector:        &StrategySelector{},
	}
	// 检测硬件能力
	db.HardwareCaps = db.strategyComputeSelector.GetHardwareCapabilities()
	logger.Info("硬件检测结果: AVX2=%v, AVX512=%v, GPU=%v, CPU核心=%d",
		db.HardwareCaps.HasAVX2, db.HardwareCaps.HasAVX512,
		db.HardwareCaps.HasGPU, db.HardwareCaps.CPUCores)

	// 如果支持GPU，初始化GPU加速器
	if db.HardwareCaps.HasGPU && db.hardwareManager != nil {
		gpuAccel := acceler.NewFAISSAccelerator(0, "Flat")

		// 先检查GPU可用性，再进行初始化
		if faissAccel, ok := gpuAccel.(*acceler.FAISSAccelerator); ok {
			if err := faissAccel.CheckGPUAvailability(); err != nil {
				logger.Warning("GPU可用性检查失败: %v", err)
				db.HardwareCaps.HasGPU = false
			} else {
				// GPU可用性检查通过，进行初始化
				if err := gpuAccel.Initialize(); err != nil {
					logger.Warning("GPU加速器初始化失败: %v", err)
					db.HardwareCaps.HasGPU = false
				} else {
					// 注册到硬件管理器
					if err := db.hardwareManager.RegisterGPUAccelerator(gpuAccel); err != nil {
						logger.Warning("注册GPU加速器失败: %v", err)
						db.HardwareCaps.HasGPU = false
					}
				}
			}
		} else {
			// 如果类型断言失败，直接尝试初始化
			if err := gpuAccel.Initialize(); err != nil {
				logger.Warning("GPU加速器初始化失败: %v", err)
				db.HardwareCaps.HasGPU = false
			} else {
				// 注册到硬件管理器
				if err := db.hardwareManager.RegisterGPUAccelerator(gpuAccel); err != nil {
					logger.Warning("注册GPU加速器失败: %v", err)
					db.HardwareCaps.HasGPU = false
				}
			}
		}
	}
	if filePath != "" {
		if err := db.LoadFromFile(filePath); err != nil {
			logger.Warning("警告: 从 %s 加载向量数据库时出错: %v。将使用空数据库启动。\n", filePath, err)
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
			db.normalizedVectors = make(map[string][]float64)
			db.compressedVectors = make(map[string]entity.CompressedVector) // 确保加载失败也初始化
		}
		db.backupPath = filePath + ".bat"
	}

	db.InitializeAdaptiveSelector()

	// 设置全局距离计算器，供其他组件使用
	setGlobalDistanceCalculator(distanceCalculator)

	return db
}

// AdaptiveCosineSimilarityBatch 自适应批量余弦相似度计算
func (db *VectorDB) AdaptiveCosineSimilarityBatch(queries [][]float64, targets [][]float64) ([][]float64, error) {
	if len(queries) == 0 || len(targets) == 0 {
		return nil, fmt.Errorf("查询向量或目标向量不能为空")
	}

	// 选择最优计算策略
	dataSize := len(queries) * len(targets)
	vectorDim := len(queries[0])
	optimalStrategy := db.strategyComputeSelector.SelectOptimalStrategy(dataSize, vectorDim)

	logger.Trace("选择计算策略: %v, 数据量: %d, 向量维度: %d", optimalStrategy, dataSize, vectorDim)

	switch optimalStrategy {
	case acceler.StrategyGPU:
		return db.gpuBatchCosineSimilarity(queries, targets)
	case acceler.StrategyAVX512, acceler.StrategyAVX2:
		return db.simdBatchCosineSimilarity(queries, targets, optimalStrategy)
	default:
		return db.standardBatchCosineSimilarity(queries, targets)
	}
}

// generateSampleQueries 生成样本查询向量
func (db *VectorDB) generateSampleQueries(dim int) [][]float64 {
	// 生成不同类型的样本查询向量
	queries := make([][]float64, 0)

	// 1. 随机向量
	for i := 0; i < 5; i++ {
		query := make([]float64, dim)
		for j := 0; j < dim; j++ {
			query[j] = rand.Float64()*2 - 1 // 生成-1到1之间的随机数
		}
		queries = append(queries, acceler.NormalizeVector(query))
	}

	// 2. 从现有数据中采样
	db.mu.RLock()
	if len(db.vectors) > 0 {
		// 随机选择一些现有向量
		samples := db.getRandomVectors(5)
		for _, sample := range samples {
			// 添加一些噪声
			noisyVector := make([]float64, len(sample))
			copy(noisyVector, sample)
			for j := 0; j < len(noisyVector); j++ {
				noisyVector[j] += (rand.Float64()*0.2 - 0.1) // 添加-0.1到0.1之间的噪声
			}
			queries = append(queries, acceler.NormalizeVector(noisyVector))
		}
	}
	db.mu.RUnlock()

	// 3. 特殊向量（全1，全0等）
	allOnes := make([]float64, dim)
	for i := 0; i < dim; i++ {
		allOnes[i] = 1.0
	}
	queries = append(queries, acceler.NormalizeVector(allOnes))

	// 稀疏向量
	sparse := make([]float64, dim)
	for i := 0; i < dim/10; i++ {
		sparse[rand.Intn(dim)] = 1.0
	}
	queries = append(queries, acceler.NormalizeVector(sparse))

	return queries
}

// getRandomVectors 从数据库中随机获取向量
func (db *VectorDB) getRandomVectors(count int) [][]float64 {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return nil
	}

	// 获取所有向量ID
	ids := make([]string, 0, len(db.vectors))
	for id := range db.vectors {
		ids = append(ids, id)
	}

	// 随机选择count个向量
	result := make([][]float64, 0, min(count, len(ids)))
	for i := 0; i < min(count, len(ids)); i++ {
		// 随机选择一个索引
		idx := rand.Intn(len(ids))
		// 获取对应的向量
		result = append(result, db.vectors[ids[idx]])
		// 从ids中移除已选择的ID（可选，避免重复选择）
		ids[idx] = ids[len(ids)-1]
		ids = ids[:len(ids)-1]
	}

	return result
}

// gpuBatchCosineSimilarity GPU批量余弦相似度计算
func (db *VectorDB) gpuBatchCosineSimilarity(queries [][]float64, targets [][]float64) ([][]float64, error) {
	// 检查GPU加速器是否初始化
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	if gpuAccelerator == nil {
		logger.Warning("GPU加速器未初始化，使用CPU计算")
		return db.standardBatchCosineSimilarity(queries, targets)
	}

	// 记录计算开始时间，用于性能分析
	startTime := time.Now()

	// 记录计算规模
	logger.Debug("开始GPU批量余弦相似度计算，查询数量: %d, 目标数量: %d", len(queries), len(targets))

	// 调用GPU加速器计算
	results, err := gpuAccelerator.BatchCosineSimilarity(queries, targets)

	// 计算耗时
	elapsed := time.Since(startTime)

	// 错误处理和回退逻辑
	if err != nil {
		// 记录详细错误信息
		logger.Warning("GPU计算失败(耗时: %v)，回退到CPU: %v", elapsed, err)

		// 检查是否是内存不足错误
		if strings.Contains(err.Error(), "memory") || strings.Contains(err.Error(), "CUDA") {
			logger.Error("GPU内存不足或CUDA错误，考虑调整批处理大小或使用CPU计算")

			// 更新统计信息
			db.statsMu.Lock()
			db.stats.GPUErrors++
			db.statsMu.Unlock()
		}

		// 回退到CPU计算
		cpuStartTime := time.Now()
		cpuResults, cpuErr := db.standardBatchCosineSimilarity(queries, targets)
		cpuElapsed := time.Since(cpuStartTime)

		if cpuErr != nil {
			logger.Error("CPU回退计算也失败: %v", cpuErr)
			return nil, fmt.Errorf("GPU和CPU计算均失败: %v, %v", err, cpuErr)
		}

		logger.Info("CPU回退计算成功，耗时: %v (GPU尝试耗时: %v)", cpuElapsed, elapsed)
		return cpuResults, nil
	}

	// 计算成功，记录性能信息
	logger.Debug("GPU批量余弦相似度计算完成，耗时: %v", elapsed)

	// 更新统计信息
	db.statsMu.Lock()
	db.stats.GPUComputations++
	db.stats.TotalGPUTime += elapsed.Milliseconds()
	db.statsMu.Unlock()

	return results, nil
}

// simdBatchCosineSimilarity SIMD批量余弦相似度计算
func (db *VectorDB) simdBatchCosineSimilarity(queries [][]float64, targets [][]float64, strategy acceler.ComputeStrategy) ([][]float64, error) {
	results := make([][]float64, len(queries))

	// 并行计算
	numWorkers := runtime.NumCPU()
	workChan := make(chan int, len(queries))
	errChan := make(chan error, 1)
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for queryIdx := range workChan {
				query := queries[queryIdx]
				similarities := make([]float64, len(targets))

				for targetIdx, target := range targets {
					similarities[targetIdx] = acceler.AdaptiveCosineSimilarity(query, target, strategy)
				}

				results[queryIdx] = similarities
			}
		}()
	}

	// 发送任务
	for i := range queries {
		workChan <- i
	}
	close(workChan)

	wg.Wait()
	close(errChan)

	// 检查错误
	select {
	case err := <-errChan:
		return nil, err
	default:
		return results, nil
	}
}

// standardBatchCosineSimilarity 标准批量余弦相似度计算
func (db *VectorDB) standardBatchCosineSimilarity(queries [][]float64, targets [][]float64) ([][]float64, error) {
	results := make([][]float64, len(queries))

	// 并行计算
	numWorkers := runtime.NumCPU()
	workChan := make(chan int, len(queries))
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for queryIdx := range workChan {
				query := queries[queryIdx]
				similarities := make([]float64, len(targets))

				for targetIdx, target := range targets {
					similarities[targetIdx] = acceler.CosineSimilarity(query, target)
				}

				results[queryIdx] = similarities
			}
		}()
	}

	// 发送任务
	for i := range queries {
		workChan <- i
	}
	close(workChan)

	wg.Wait()

	return results, nil
}

// AdaptiveFindNearest 自适应最近邻搜索
func (db *VectorDB) AdaptiveFindNearest(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("vectors is empty")
	}

	// 选择最优计算策略
	dataSize := len(db.vectors)
	vectorDim := len(query)
	optimalStrategy := db.strategyComputeSelector.SelectOptimalStrategy(dataSize, vectorDim)

	logger.Trace("自适应搜索策略: %v", optimalStrategy)

	// 如果启用了HNSW索引，优先使用
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		return db.hnswAdaptiveSearch(query, k, optimalStrategy)
	}

	// 使用IVF索引进行自适应搜索
	return db.ivfAdaptiveSearch(query, k, nprobe, optimalStrategy)
}

// hnswAdaptiveSearch HNSW自适应搜索
func (db *VectorDB) hnswAdaptiveSearch(query []float64, k int, strategy acceler.ComputeStrategy) ([]entity.Result, error) {
	// 根据策略设置距离函数
	switch strategy {
	case acceler.StrategyAVX512, acceler.StrategyAVX2:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := acceler.AdaptiveCosineSimilarity(a, b, strategy)
			return 1.0 - sim, nil
		})
	default:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := acceler.CosineSimilarity(a, b)
			return 1.0 - sim, nil
		})
	}

	// 执行搜索
	normalizedQuery := acceler.NormalizeVector(query)
	results, err := db.hnsw.Search(normalizedQuery, k)
	if err != nil {
		return nil, err
	}

	return results, nil
}

// ivfAdaptiveSearch IVF自适应搜索
func (db *VectorDB) ivfAdaptiveSearch(query []float64, k int, nprobe int, strategy acceler.ComputeStrategy) ([]entity.Result, error) {
	if !db.indexed {
		return nil, fmt.Errorf("数据库尚未建立索引")
	}

	// 粗排：找到最近的nprobe个簇
	candidateClusters := make([]int, 0, nprobe)
	clusterDistances := make([]float64, len(db.clusters))

	for i, cluster := range db.clusters {
		// 使用自适应距离计算
		switch strategy {
		case acceler.StrategyAVX512, acceler.StrategyAVX2:
			sim := acceler.AdaptiveCosineSimilarity(query, cluster.Centroid, strategy)
			clusterDistances[i] = 1.0 - sim
		default:
			sim := acceler.CosineSimilarity(query, cluster.Centroid)
			clusterDistances[i] = 1.0 - sim
		}
	}

	// 选择距离最近的nprobe个簇
	type clusterDist struct {
		index    int
		distance float64
	}

	clusterList := make([]clusterDist, len(db.clusters))
	for i, dist := range clusterDistances {
		clusterList[i] = clusterDist{index: i, distance: dist}
	}

	sort.Slice(clusterList, func(i, j int) bool {
		return clusterList[i].distance < clusterList[j].distance
	})

	for i := 0; i < nprobe && i < len(clusterList); i++ {
		candidateClusters = append(candidateClusters, clusterList[i].index)
	}

	// 精排：在候选簇中搜索最近邻
	candidateVectors := make([]string, 0)
	for _, clusterIdx := range candidateClusters {
		candidateVectors = append(candidateVectors, db.clusters[clusterIdx].VectorIDs...)
	}

	return db.adaptiveFineRanking(query, candidateVectors, k, strategy)
}

// adaptiveFineRanking 自适应精排
func (db *VectorDB) adaptiveFineRanking(query []float64, candidates []string, k int, strategy acceler.ComputeStrategy) ([]entity.Result, error) {
	if len(candidates) == 0 {
		return nil, fmt.Errorf("candidates is empty")
	}

	// 根据策略选择计算方法
	switch strategy {
	case acceler.StrategyGPU:
		return db.gpuFineRanking(query, candidates, k)
	default:
		return db.cpuFineRanking(query, candidates, k, strategy)
	}
}

// cpuFineRanking CPU精排（支持SIMD加速）
func (db *VectorDB) cpuFineRanking(query []float64, candidates []string, k int, strategy acceler.ComputeStrategy) ([]entity.Result, error) {

	results := make([]entity.Result, 0, len(candidates))
	resultsChan := make(chan entity.Result, len(candidates))

	// 并行计算相似度
	numWorkers := runtime.NumCPU()
	workChan := make(chan string, len(candidates))
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for candidateID := range workChan {
				if vec, exists := db.vectors[candidateID]; exists {
					// 使用自适应相似度计算
					sim := acceler.AdaptiveCosineSimilarity(query, vec, strategy)
					resultsChan <- entity.Result{Id: candidateID, Similarity: sim}
				}
			}
		}()
	}

	// 发送任务
	for _, candidateID := range candidates {
		workChan <- candidateID
	}
	close(workChan)

	wg.Wait()
	close(resultsChan)

	// 收集结果
	for result := range resultsChan {
		results = append(results, result)
	}

	// 排序并返回top-k
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	if len(results) > k {
		results = results[:k]
	}
	return results, nil
}

// gpuFineRanking GPU加速的精排方法
func (db *VectorDB) gpuFineRanking(query []float64, candidateIDs []string, k int) ([]entity.Result, error) {
	// 获取GPU加速器
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	if gpuAccelerator == nil {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	// 构建候选向量数组
	candidateVectors := make([][]float64, 0, len(candidateIDs))
	validIDs := make([]string, 0, len(candidateIDs))

	for _, id := range candidateIDs {
		if vec, exists := db.vectors[id]; exists {
			candidateVectors = append(candidateVectors, vec)
			validIDs = append(validIDs, id)
		}
	}

	if len(candidateVectors) == 0 {
		return []entity.Result{}, nil
	}

	// 使用GPU批量搜索进行精排
	queries := [][]float64{query}
	gpuResults, err := gpuAccelerator.BatchSearch(queries, candidateVectors, k)
	if err != nil {
		return nil, fmt.Errorf("GPU精排失败: %w", err)
	}

	if len(gpuResults) == 0 || len(gpuResults[0]) == 0 {
		return []entity.Result{}, nil
	}

	// 转换结果
	results := make([]entity.Result, 0, min(k, len(gpuResults[0])))
	for _, gpuResult := range gpuResults[0] {
		if idx, err := strconv.Atoi(gpuResult.ID); err == nil && idx < len(validIDs) {
			results = append(results, entity.Result{
				Id:         validIDs[idx],
				Similarity: gpuResult.Similarity,
			})
		}
	}

	return results, nil
}

// SetComputeStrategy 手动设置计算策略
func (db *VectorDB) SetComputeStrategy(strategy acceler.ComputeStrategy) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 验证策略是否可用
	switch strategy {
	case acceler.StrategyAVX2:
		if !db.HardwareCaps.HasAVX2 {
			return fmt.Errorf("当前硬件不支持AVX2指令集")
		}
	case acceler.StrategyAVX512:
		if !db.HardwareCaps.HasAVX512 {
			return fmt.Errorf("当前硬件不支持AVX512指令集")
		}
	case acceler.StrategyGPU:
		if !db.HardwareCaps.HasGPU {
			return fmt.Errorf("当前系统不支持GPU加速")
		}
	default:
		return fmt.Errorf("unhandled default case")
	}

	db.currentStrategy = strategy
	logger.Info("手动设置计算策略为: %v", strategy)
	return nil
}

// GetHardwareInfo 获取硬件信息
func (db *VectorDB) GetHardwareInfo() acceler.HardwareCapabilities {
	return db.HardwareCaps
}

// GetCurrentStrategy 获取当前计算策略
func (db *VectorDB) GetCurrentStrategy() acceler.ComputeStrategy {
	return db.currentStrategy
}

// GetSelectStrategy 动态选择最佳计算策略
func (db *VectorDB) GetSelectStrategy() acceler.ComputeStrategy {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 优先选择最高性能的可用策略
	if db.HardwareCaps.HasAVX512 {
		return acceler.StrategyAVX512
	} else if db.HardwareCaps.HasAVX2 {
		return acceler.StrategyAVX2
	}
	return acceler.StrategyStandard
}

// BatchAddToHNSWIndex 批量添加向量到 HNSW 索引
// 当需要添加大量向量时，使用此方法比单个添加更高效
// numWorkers: 并行工作的协程数量，如果 <= 0，则使用 CPU 核心数
func (db *VectorDB) BatchAddToHNSWIndex(ids []string, vectors [][]float64, numWorkers int) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if !db.useHNSWIndex || db.hnsw == nil {
		return fmt.Errorf("HNSW 索引未启用或未初始化")
	}

	if len(ids) != len(vectors) {
		return fmt.Errorf("ID数量与向量数量不匹配: %d != %d", len(ids), len(vectors))
	}

	// 如果启用了向量归一化，预处理向量
	processedVectors := make([][]float64, len(vectors))
	for i, vec := range vectors {
		if db.useNormalization {
			processedVectors[i] = acceler.NormalizeVector(vec)
		} else {
			processedVectors[i] = vec
		}
	}

	// 使用 HNSW 图的并行添加节点方法
	startTime := time.Now()
	logger.Info("开始批量添加 %d 个向量到 HNSW 索引...", len(ids))

	err := db.hnsw.ParallelAddNodes(ids, processedVectors, numWorkers)
	if err != nil {
		return fmt.Errorf("批量添加节点失败: %w", err)
	}

	logger.Info("成功批量添加 %d 个向量到 HNSW 索引，耗时 %v", len(ids), time.Since(startTime))
	return nil
}

// GetAllIDs 返回数据库中所有向量的ID列表
func (db *VectorDB) GetAllIDs() []string {
	db.mu.RLock()
	defer db.mu.RUnlock()

	ids := make([]string, 0, len(db.vectors))
	for id := range db.vectors {
		ids = append(ids, id)
	}

	return ids
}

// BuildHNSWIndexParallel 并行构建 HNSW 图结构索引
// 这是 BuildHNSWIndex 的并行版本，适用于大规模数据集
func (db *VectorDB) BuildHNSWIndexParallel(numWorkers int) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	startTime := time.Now()
	logger.Info("开始并行构建 HNSW 索引...")

	// 重置索引状态
	db.indexed = false

	// 创建新的 HNSW 图
	db.hnsw = graph.NewHNSWGraph(db.maxConnections, db.efConstruction, db.efSearch)

	// 设置距离函数
	db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
		// 使用余弦距离（1 - 余弦相似度）
		sim := acceler.CosineSimilarity(a, b)
		return 1.0 - sim, nil
	})

	// 准备批量添加的数据
	ids := make([]string, 0, len(db.vectors))
	vectors := make([][]float64, 0, len(db.vectors))

	for id, vec := range db.vectors {
		// 如果启用了向量归一化，使用归一化后的向量
		vector := vec
		if db.useNormalization {
			if normalizedVec, exists := db.normalizedVectors[id]; exists {
				vector = normalizedVec
			}
		}

		ids = append(ids, id)
		vectors = append(vectors, vector)
	}

	// 使用并行方法添加所有节点
	err := db.hnsw.ParallelAddNodes(ids, vectors, numWorkers)
	if err != nil {
		return fmt.Errorf("并行添加节点到 HNSW 图失败: %w", err)
	}

	db.indexed = true
	db.stats.IndexBuildTime = time.Since(startTime)
	db.stats.LastReindexTime = time.Now()

	logger.Info("HNSW 索引并行构建完成，耗时 %v，包含 %d 个向量。", db.stats.IndexBuildTime, len(db.vectors))
	return nil
}

// BatchFindNearest 批量查找多个查询向量的最近邻
// queryVectors: 查询向量数组
// k: 每个查询返回的最近邻数量
// numWorkers: 并行工作的协程数量，如果 <= 0，则使用 CPU 核心数
// 返回: 每个查询向量对应的最近邻ID数组，以及可能的错误
func (db *VectorDB) BatchFindNearest(queryVectors [][]float64, k int, numWorkers int) ([][]entity.Result, error) {
	startTime := time.Now()

	// 更新查询计数
	db.statsMu.Lock()
	db.stats.TotalQueries += int64(len(queryVectors))
	db.statsMu.Unlock()

	db.mu.RLock()
	defer db.mu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		// 如果数据库为空，返回空结果
		emptyResults := make([][]entity.Result, len(queryVectors))
		for i := range emptyResults {
			emptyResults[i] = []entity.Result{}
		}
		return emptyResults, fmt.Errorf("vector is empty")
	}

	// 如果启用了 HNSW 索引，使用 HNSW 批量搜索
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		logger.Trace("使用 HNSW 索引进行批量搜索，查询数量: %d", len(queryVectors))

		// 预处理查询向量（归一化）
		normalizedQueries := make([][]float64, len(queryVectors))
		for i, query := range queryVectors {
			normalizedQueries[i] = acceler.NormalizeVector(query)
		}

		// 使用 HNSW 批量搜索
		batchResults, err := db.hnsw.BatchSearch(normalizedQueries, k, numWorkers)
		if err != nil {
			return nil, fmt.Errorf("HNSW 批量搜索失败: %w", err)
		}

		// 提取 ID
		results := make([][]entity.Result, len(batchResults))
		for i, queryResult := range batchResults {
			ids := make([]entity.Result, min(k, len(queryResult)))
			for j, result := range queryResult {
				ids[j] = result
			}
			results[i] = ids
		}

		// 更新平均查询时间统计
		queryTime := time.Since(startTime)
		db.statsMu.Lock()
		avgTime := time.Duration(queryTime.Nanoseconds() / int64(len(queryVectors)))
		db.stats.AvgQueryTime = time.Duration((db.stats.AvgQueryTime.Nanoseconds()*9 + avgTime.Nanoseconds()) / 10) // 使用加权平均
		db.statsMu.Unlock()

		return results, nil
	}

	// 如果未启用 HNSW 索引，使用并行的暴力搜索或其他索引方法
	results := make([][]entity.Result, len(queryVectors))
	errChan := make(chan error, len(queryVectors))

	// 使用信号量限制并发数
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	sem := make(chan struct{}, numWorkers)
	var wg sync.WaitGroup

	for i, query := range queryVectors {
		wg.Add(1)
		go func(idx int, q []float64) {
			defer wg.Done()

			// 获取信号量
			sem <- struct{}{}
			defer func() { <-sem }()

			// 使用单个查询方法
			ids, err := db.FindNearest(q, k, db.numClusters/10) // 使用默认的 nprobe 值
			if err != nil {
				errChan <- fmt.Errorf("查询向量 %d 搜索失败: %w", idx, err)
				return
			}
			results[idx] = ids
		}(i, query)
	}

	wg.Wait()
	close(errChan)

	// 检查是否有错误
	select {
	case err := <-errChan:
		return nil, err
	default:
		// 更新平均查询时间统计
		queryTime := time.Since(startTime)
		db.statsMu.Lock()
		avgTime := time.Duration(queryTime.Nanoseconds() / int64(len(queryVectors)))
		db.stats.AvgQueryTime = time.Duration((db.stats.AvgQueryTime.Nanoseconds()*9 + avgTime.Nanoseconds()) / 10) // 使用加权平均
		db.statsMu.Unlock()

		return results, nil
	}
}

// BatchFindNearestWithScores 批量查找多个查询向量的最近邻，并返回相似度分数
// queryVectors: 查询向量数组
// k: 每个查询返回的最近邻数量
// numWorkers: 并行工作的协程数量，如果 <= 0，则使用 CPU 核心数
// 返回: 每个查询向量对应的最近邻结果（包含ID和相似度分数），以及可能的错误
func (db *VectorDB) BatchFindNearestWithScores(queryVectors [][]float64, k int, numWorkers int) ([][]entity.Result, error) {
	startTime := time.Now()

	// 更新查询计数
	db.statsMu.Lock()
	db.stats.TotalQueries += int64(len(queryVectors))
	db.statsMu.Unlock()

	db.mu.RLock()
	defer db.mu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		// 如果数据库为空，返回空结果
		emptyResults := make([][]entity.Result, len(queryVectors))
		for i := range emptyResults {
			emptyResults[i] = []entity.Result{}
		}
		return emptyResults, nil
	}

	// 如果启用了 HNSW 索引，使用 HNSW 批量搜索
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		logger.Trace("使用 HNSW 索引进行批量搜索（带分数），查询数量: %d", len(queryVectors))

		// 预处理查询向量（归一化）
		normalizedQueries := make([][]float64, len(queryVectors))
		for i, query := range queryVectors {
			normalizedQueries[i] = acceler.NormalizeVector(query)
		}

		// 使用 HNSW 批量搜索
		results, err := db.hnsw.BatchSearch(normalizedQueries, k, numWorkers)
		if err != nil {
			return nil, fmt.Errorf("HNSW 批量搜索失败: %w", err)
		}

		// 更新平均查询时间统计
		queryTime := time.Since(startTime)
		db.statsMu.Lock()
		avgTime := time.Duration(queryTime.Nanoseconds() / int64(len(queryVectors)))
		db.stats.AvgQueryTime = time.Duration((db.stats.AvgQueryTime.Nanoseconds()*9 + avgTime.Nanoseconds()) / 10) // 使用加权平均
		db.statsMu.Unlock()

		return results, nil
	}

	// 如果未启用 HNSW 索引，使用并行的暴力搜索或其他索引方法
	results := make([][]entity.Result, min(k, len(queryVectors)))
	errChan := make(chan error, len(queryVectors))

	// 使用信号量限制并发数
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	sem := make(chan struct{}, numWorkers)
	var wg sync.WaitGroup

	for i, query := range queryVectors {
		wg.Add(1)
		go func(idx int, q []float64) {
			defer wg.Done()

			// 获取信号量
			sem <- struct{}{}
			defer func() { <-sem }()

			// 使用单个查询方法
			resultsWithScores, err := db.FindNearestWithScores(q, k, db.numClusters/10) // 使用默认的 nprobe 值
			if err != nil {
				errChan <- fmt.Errorf("查询向量 %d 搜索失败: %w", idx, err)
				return
			}
			results[idx] = resultsWithScores
		}(i, query)
	}

	wg.Wait()
	close(errChan)

	// 检查是否有错误
	select {
	case err := <-errChan:
		return nil, err
	default:
		// 更新平均查询时间统计
		queryTime := time.Since(startTime)
		db.statsMu.Lock()
		avgTime := time.Duration(queryTime.Nanoseconds() / int64(len(queryVectors)))
		db.stats.AvgQueryTime = time.Duration((db.stats.AvgQueryTime.Nanoseconds()*9 + avgTime.Nanoseconds()) / 10) // 使用加权平均
		db.statsMu.Unlock()

		return results, nil
	}
}

// EnableHNSWIndex 启用 HNSW 索引并设置相关参数
func (db *VectorDB) EnableHNSWIndex(maxConnections int, efConstruction, efSearch float64) {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.useHNSWIndex = true
	db.maxConnections = maxConnections
	db.efConstruction = efConstruction
	db.efSearch = efSearch

	// 如果已有向量数据，立即构建索引
	if len(db.vectors) > 0 {
		db.indexed = false
		logger.Info("启用 HNSW 索引，需要重建索引。请调用 BuildIndex() 方法。")
	}
}

// BuildHNSWIndex 构建 HNSW 图结构索引
func (db *VectorDB) BuildHNSWIndex() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	startTime := time.Now()
	logger.Info("开始构建 HNSW 索引...")

	// 重置索引状态
	db.indexed = false

	// 创建新的 HNSW 图
	db.hnsw = graph.NewHNSWGraph(db.maxConnections, db.efConstruction, db.efSearch)

	// 设置距离函数
	db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
		// 使用余弦距离（1 - 余弦相似度）
		sim := acceler.CosineSimilarity(a, b)
		return 1.0 - sim, nil
	})

	// 添加所有向量到图中
	for id, vec := range db.vectors {
		// 如果启用了向量归一化，使用归一化后的向量
		vector := vec
		if db.useNormalization {
			if normalizedVec, exists := db.normalizedVectors[id]; exists {
				vector = normalizedVec
			}
		}

		err := db.hnsw.AddNode(id, vector)
		if err != nil {
			return fmt.Errorf("添加向量 %s 到 HNSW 图失败: %w", id, err)
		}
	}

	db.indexed = true
	db.stats.IndexBuildTime = time.Since(startTime)
	db.stats.LastReindexTime = time.Now()

	logger.Info("HNSW 索引构建完成，耗时 %v，包含 %d 个向量。", db.stats.IndexBuildTime, len(db.vectors))
	return nil
}

// GetVectorDimension 返回数据库中向量的维度
// 如果尚未添加任何向量或维度未初始化，则返回 0
func (db *VectorDB) GetVectorDimension() (int, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.vectorDim == 0 && len(db.vectors) > 0 {
		// 如果 vectorDim 未初始化，但存在向量，尝试从第一个向量获取维度
		// 这是一个后备逻辑，理想情况下 vectorDim 应该在添加第一个向量时设置
		for _, v := range db.vectors {
			db.mu.RUnlock() // 释放读锁，准备获取写锁
			db.mu.Lock()
			db.vectorDim = len(v) // 设置维度
			db.mu.Unlock()
			db.mu.RLock() // 重新获取读锁
			logger.Info("Vector dimension was not initialized, inferred as %d from existing vectors.", db.vectorDim)
			break
		}
	}
	if db.vectorDim == 0 {
		return 0, fmt.Errorf("向量维度尚未初始化，数据库中可能没有向量")
	}
	return db.vectorDim, nil
}

// GetTrainingVectors 从数据库中获取用于训练的向量样本
// sampleRate: 采样率 (0.0 to 1.0)
// maxVectors: 最大采样向量数 (0 表示无限制，除非 sampleRate 也为0)
func (db *VectorDB) GetTrainingVectors(sampleRate float64, maxVectors int) ([][]float64, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("数据库中没有向量可供训练")
	}

	var sampledVectors [][]float64
	allVectorIDs := make([]string, 0, len(db.vectors))
	for id := range db.vectors {
		allVectorIDs = append(allVectorIDs, id)
	}

	rand.Shuffle(len(allVectorIDs), func(i, j int) {
		allVectorIDs[i], allVectorIDs[j] = allVectorIDs[j], allVectorIDs[i]
	})

	numToSample := 0
	if sampleRate > 0 {
		numToSample = int(float64(len(db.vectors)) * sampleRate)
	}

	if maxVectors > 0 {
		if numToSample == 0 || numToSample > maxVectors { // 如果采样数超过最大数，或未通过采样率设置
			numToSample = maxVectors
		}
	}

	if numToSample == 0 { // 如果两者都未有效设置，则默认采样一小部分或全部（如果数据量小）
		numToSample = len(db.vectors)
		if numToSample > 10000 { // 避免采样过多数据，设定一个上限
			numToSample = 10000
		}
	}

	if numToSample > len(allVectorIDs) {
		numToSample = len(allVectorIDs)
	}

	sampledVectors = make([][]float64, 0, numToSample)
	for i := 0; i < numToSample; i++ {
		id := allVectorIDs[i]
		// 需要复制一份，避免外部修改影响原始数据
		vecCopy := make([]float64, len(db.vectors[id]))
		copy(vecCopy, db.vectors[id])
		sampledVectors = append(sampledVectors, vecCopy)
	}

	logger.Info("从 VectorDB 采样了 %d 个向量用于训练", len(sampledVectors))
	return sampledVectors, nil
}

// AddMetadata 在VectorDB结构体中添加元数据支持
func (db *VectorDB) AddMetadata(id string, metadata map[string]interface{}) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.metadata == nil {
		db.metadata = make(map[string]map[string]interface{})
	}

	db.metadata[id] = metadata
}

// GetMetadata 获取文档元数据
func (db *VectorDB) GetMetadata(id string) (map[string]interface{}, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.metadata == nil {
		return nil, false
	}

	metadata, exists := db.metadata[id]
	return metadata, exists
}

// SearchWithFilter 带过滤条件的向量搜索
func (db *VectorDB) SearchWithFilter(query string, topK int, filter func(map[string]interface{}) bool) ([]entity.SearchResult, error) {
	// 将查询文本向量化
	queryVector, err := db.GetVectorForTextWithCache(query, db.vectorizedType)
	if err != nil {
		return nil, fmt.Errorf("查询文本向量化失败: %v", err)
	}

	// 查找最近的向量
	results, err := db.FindNearest(queryVector, topK*2, 10) // 获取更多结果用于过滤
	if err != nil {
		return nil, fmt.Errorf("搜索失败: %v", err)
	}

	// 应用过滤器并计算相似度
	var filteredResults []entity.SearchResult
	for _, result := range results {
		// 获取元数据
		metadata, exists := db.GetMetadata(result.Id)
		if !exists {
			metadata = make(map[string]interface{})
		}

		// 应用过滤器
		if filter == nil || filter(metadata) {
			// 计算相似度
			similarity, err := db.CalculateCosineSimilarity(result.Id, queryVector)
			if err != nil {
				continue
			}

			filteredResults = append(filteredResults, entity.SearchResult{
				ID:         result.Id,
				Similarity: similarity,
				Metadata:   metadata,
			})
		}

		// 如果已经有足够的结果，停止处理
		if len(filteredResults) >= topK {
			break
		}
	}

	// 按相似度排序
	sort.Slice(filteredResults, func(i, j int) bool {
		return filteredResults[i].Similarity > filteredResults[j].Similarity
	})

	// 限制结果数量
	if len(filteredResults) > topK {
		filteredResults = filteredResults[:topK]
	}

	return filteredResults, nil
}

// CalculateCosineSimilarity 计算指定ID的向量与查询向量之间的余弦相似度
func (db *VectorDB) CalculateCosineSimilarity(id string, queryVector []float64) (float64, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 获取指定ID的向量
	vector, exists := db.vectors[id]
	if !exists {
		return 0, fmt.Errorf("向量ID %s 不存在", id)
	}

	// 如果启用了向量归一化，优先使用归一化后的向量
	if db.useNormalization {
		if normalizedVec, exists := db.normalizedVectors[id]; exists {
			vector = normalizedVec
		}
	}

	// 归一化查询向量
	normalizedQuery := queryVector
	if db.useNormalization {
		normalizedQuery = acceler.NormalizeVector(queryVector)
	}

	// 计算余弦相似度
	similarity := acceler.CosineSimilarity(normalizedQuery, vector)
	if similarity < 0 {
		return 0, fmt.Errorf("计算余弦相似度失败：向量维度不匹配")
	}

	return similarity, nil
}

// EnablePQCompression 启用 PQ 压缩并设置相关参数
// codebookPath: 码本文件路径。如果为空，则尝试使用之前配置的路径加载，或禁用PQ。
// numSubVectors, numCentroidsPerSubVector: 这些参数现在主要用于信息展示和潜在的校验，实际值会从加载的码本中推断。
func (db *VectorDB) EnablePQCompression(codebookPath string, numSubVectors int, numCentroidsPerSubVector int) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 如果传入了新的 codebookPath，则使用它
	// 如果 codebookPath 为空，则尝试使用 db.pqCodebookFilePath (如果之前设置过)
	pathToLoad := codebookPath
	if pathToLoad == "" {
		pathToLoad = db.pqCodebookFilePath
	}

	if pathToLoad == "" {
		db.mu.Lock()
		logger.Warning("未提供 PQ 码本文件路径，且之前未配置，PQ 压缩将禁用。")
		db.usePQCompression = false
		db.pqCodebook = nil
		db.numSubVectors = 0
		db.numCentroidsPerSubVector = 0
		db.mu.Unlock()
		return nil // 不是错误，只是禁用
	}
	if numSubVectors <= 0 {
		return fmt.Errorf("子向量数量必须为正")
	}
	if numCentroidsPerSubVector <= 0 {
		return fmt.Errorf("每个子向量的质心数量必须为正")
	}

	if err := db.LoadPQCodebookFromFile(pathToLoad); err != nil {
		// LoadPQCodebookFromFile 内部会处理文件不存在的情况并禁用PQ，这里只处理其他加载错误
		db.mu.Lock()
		db.usePQCompression = false // 加载失败，禁用PQ
		db.pqCodebook = nil
		db.numSubVectors = 0
		db.numCentroidsPerSubVector = 0
		db.mu.Unlock()
		return fmt.Errorf("启用 PQ 压缩失败，加载码本时出错: %v", err)
	}
	db.mu.Lock() // 确保在更新 usePQCompression 之前获取锁
	// 只有当码本成功加载且非空时，才真正启用PQ
	if db.pqCodebook != nil && len(db.pqCodebook) > 0 {
		db.usePQCompression = true
		// 更新 numSubVectors 和 numCentroidsPerSubVector 以匹配加载的码本
		db.numSubVectors = len(db.pqCodebook)
		if len(db.pqCodebook[0]) > 0 {
			db.numCentroidsPerSubVector = len(db.pqCodebook[0])
		} else {
			db.numCentroidsPerSubVector = 0 // 或者报错，取决于策略
		}
		logger.Info("PQ 压缩已启用。码本路径: %s, 子向量数: %d, 每子空间质心数: %d", db.pqCodebookFilePath, db.numSubVectors, db.numCentroidsPerSubVector)

		// 提示用户可能需要压缩现有向量
		if len(db.vectors) > 0 && len(db.compressedVectors) < len(db.vectors) {
			err := db.CompressExistingVectors()
			if err != nil {
				return err
			}
		}
	} else {
		db.usePQCompression = false
		logger.Warning("PQ 码本加载后为空或加载失败，PQ 压缩已禁用。")
	}
	db.mu.Unlock()

	return nil
}

// CompressExistingVectors 对数据库中所有尚未压缩的向量进行 PQ 压缩
func (db *VectorDB) CompressExistingVectors() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if !db.usePQCompression || db.pqCodebook == nil {
		return fmt.Errorf("PQ 压缩未启用或码本未设置")
	}

	logger.Info("开始压缩现有向量...")
	// 收集需要压缩的向量
	vectorsToCompress := make([][]float64, 0)
	idsToCompress := make([]string, 0)

	for id, vec := range db.vectors {
		if _, exists := db.compressedVectors[id]; !exists {
			vectorsToCompress = append(vectorsToCompress, vec)
			idsToCompress = append(idsToCompress, id)
		}
	}

	totalVectors := len(vectorsToCompress)
	if totalVectors == 0 {
		logger.Info("没有需要压缩的向量。")
		return nil
	}

	logger.Info("发现 %d 个未压缩向量，开始批量压缩处理...", totalVectors)

	// 使用批量压缩函数
	numWorkers := runtime.NumCPU() // 使用所有可用CPU核心
	logger.Info("使用 %d 个工作协程进行并行压缩", numWorkers)

	startTime := time.Now()
	compressedVectors, err := acceler.BatchCompressByPQ(
		vectorsToCompress,
		db.pqCodebook,
		db.numSubVectors,
		db.numCentroidsPerSubVector,
		numWorkers,
	)

	if err != nil {
		logger.Error("批量压缩向量失败: %v", err)
		return fmt.Errorf("批量压缩向量失败: %w", err)
	}

	// 将压缩结果存储到数据库
	for i, id := range idsToCompress {
		db.compressedVectors[id] = compressedVectors[i]
	}

	elapsedTime := time.Since(startTime)
	logger.Info("现有向量批量压缩完成，共压缩了 %d 个向量，耗时 %v，平均每个向量 %.2f 毫秒。",
		totalVectors,
		elapsedTime,
		float64(elapsedTime.Milliseconds())/float64(totalVectors))

	// 更新压缩状态
	db.useCompression = true

	return nil
}

func (db *VectorDB) GetVectors() map[string][]float64 {
	return db.vectors
}

// 优化7: 添加向量预处理函数
func (db *VectorDB) preprocessVector(id string, vector []float64) {
	// 更新向量维度
	if db.vectorDim == 0 && len(vector) > 0 {
		db.vectorDim = len(vector)
	}

	// 预计算并存储归一化向量
	db.normalizedVectors[id] = acceler.NormalizeVector(vector)
}

// AddDocument 添加文档并将其转换为向量后存入数据库
func (db *VectorDB) AddDocument(id string, doc string, vectorizedType int) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	vector, err := db.GetVectorForTextWithCache(doc, vectorizedType) // Use GetVectorForText internally
	if err != nil {
		return fmt.Errorf("failed to vectorize document %s for AddDocument: %w", id, err)
	}

	// 设置向量维度（如果尚未设置）
	if db.vectorDim == 0 && len(vector) > 0 {
		db.vectorDim = len(vector)
	} else if db.vectorDim > 0 && len(vector) != db.vectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d for document %s", db.vectorDim, len(vector), id)
	}

	// 将向量添加到数据库
	db.vectors[id] = vector
	// 预计算并存储归一化向量
	db.normalizedVectors[id] = acceler.NormalizeVector(vector)
	// 如果启用了 PQ 压缩，则压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := acceler.OptimizedCompressByPQ(vector, db.pqCodebook, db.numSubVectors, db.numCentroidsPerSubVector)
		if err != nil {
			// 即使压缩失败，原始向量也已添加，这里只记录错误
			logger.Error("为文档 %s 添加时压缩向量失败: %v", id, err)
		} else {
			db.compressedVectors[id] = compressedVec
		}
	}
	// 更新倒排索引
	db.invertedMu.Lock()
	words := strings.Fields(doc)
	for _, word := range words {
		if _, exists := db.invertedIndex[word]; !exists {
			db.invertedIndex[word] = make([]string, 0, 1)
		}
		// 检查ID是否已存在，避免重复
		found := false
		for _, existingID := range db.invertedIndex[word] {
			if existingID == id {
				found = true
				break
			}
		}
		if !found {
			db.invertedIndex[word] = append(db.invertedIndex[word], id)
		}
	}
	db.invertedMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		logger.Info("提示: 添加新文档向量后，索引已失效，请重新调用 BuildIndex()。")
	}
	return nil
}

// RebuildIndex 重建向量数据库索引
func (db *VectorDB) RebuildIndex() error {
	// 记录开始时间，用于性能统计
	start := time.Now()
	// 如果启用了 HNSW 索引，使用 HNSW 构建方法
	if db.useHNSWIndex {
		return db.BuildHNSWIndex()
	}
	// 清除旧索引
	db.mu.Lock()
	db.indexed = false
	db.clusters = make([]Cluster, 0)
	db.mu.Unlock()

	// 设置索引状态为未索引
	db.indexed = false
	// 使用默认参数重建索引
	// 最大迭代次数设为100，收敛容差设为0.001
	err := db.BuildIndex(100, 0.001)

	// 更新性能统计信息
	if err != nil {
		logger.Error("索引重建失败: %v", err)
	}
	db.statsMu.Lock()
	db.stats.IndexBuildTime = time.Since(start)
	db.stats.LastReindexTime = time.Now()

	// 更新内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	db.stats.MemoryUsage = m.Alloc
	db.statsMu.Unlock()
	logger.Info("索引重建完成，耗时: %v", time.Since(start))

	return err
}

// RebuildIndexWithType 重建向量数据库索引
func (db *VectorDB) RebuildIndexWithType(indexType string, indexParams map[string]interface{}) error {
	// 记录开始时间，用于性能统计
	start := time.Now()
	logger.Info("开始重建索引，类型: %s, 参数: %v", indexType, indexParams)

	var err error
	db.mu.Lock()
	db.indexed = false               // 标记索引失效
	db.clusters = make([]Cluster, 0) // 清空旧的IVF簇
	// 如果存在HNSW索引实例，也需要考虑重置或重建
	if db.hnsw != nil {
		// 根据策略，可以选择关闭旧实例并新建，或者尝试更新参数（如果支持）
		// 这里简单地置为nil，强制在BuildHNSWIndex中重新初始化
		db.hnsw = nil
		logger.Info("HNSW索引实例已清除，将在重建时重新初始化")
	}
	db.mu.Unlock()

	switch indexType {
	case "HNSW":
		db.mu.Lock()
		db.useHNSWIndex = true
		// 从 indexParams 解析 HNSW 特定参数
		if m, ok := indexParams["maxConnections"].(float64); ok {
			db.maxConnections = int(m)
		}
		if efc, ok := indexParams["efConstruction"].(float64); ok {
			db.efConstruction = efc
		}
		if efs, ok := indexParams["efSearch"].(float64); ok {
			db.efSearch = efs
		}
		db.mu.Unlock()
		err = db.BuildHNSWIndex() // BuildHNSWIndex 内部会使用更新后的参数
	case "EnhancedIVF":
		db.mu.Lock()
		db.useHNSWIndex = false // 确保不使用HNSW
		// 假设 EnhancedIVF 使用 BuildIndex，并可能需要从 indexParams 获取参数
		// 例如：numClusters, maxIterations, tolerance
		numClusters := db.numClusters // 默认值
		maxIterations := 100          // 默认值
		tolerance := 0.001            // 默认值
		if nc, ok := indexParams["numClusters"].(float64); ok {
			numClusters = int(nc)
		}
		if mi, ok := indexParams["maxIterations"].(float64); ok {
			maxIterations = int(mi)
		}
		if tol, ok := indexParams["tolerance"].(float64); ok {
			tolerance = tol
		}
		db.numClusters = numClusters // 更新db的配置
		db.mu.Unlock()
		err = db.BuildIndex(maxIterations, tolerance)
	case "EnhancedLSH", "IVF": // 假设这些也使用 BuildIndex 或有特定构建方法
		// 此处为简化，也指向BuildIndex，实际应有各自的逻辑
		db.mu.Lock()
		db.useHNSWIndex = false
		numClusters := db.numClusters
		maxIterations := 100
		tolerance := 0.001
		if nc, ok := indexParams["numClusters"].(float64); ok {
			numClusters = int(nc)
		}
		db.numClusters = numClusters
		db.mu.Unlock()
		err = db.BuildIndex(maxIterations, tolerance)
	default:
		err = fmt.Errorf("不支持的索引类型: %s", indexType)
	}

	// 更新性能统计信息
	if err != nil {
		logger.Error("索引重建失败: %v", err)
	} else {
		db.mu.Lock()
		db.indexed = true // 标记索引构建成功
		db.mu.Unlock()
		logger.Info("索引重建成功完成")
	}

	db.statsMu.Lock()
	db.stats.IndexBuildTime = time.Since(start)
	db.stats.LastReindexTime = time.Now()

	// 更新内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	db.stats.MemoryUsage = m.Alloc
	db.statsMu.Unlock()
	logger.Info("索引重建过程结束，耗时: %v", time.Since(start))

	return err
}

// GetVectorForTextWithCache 带缓存的向量获取
func (db *VectorDB) GetVectorForTextWithCache(text string, vectorizedType int) ([]float64, error) {
	// 生成缓存键
	cacheKey := fmt.Sprintf("vector:%d:%s", vectorizedType, text)

	// 检查MultiCache
	if db.MultiCache != nil {
		if cachedVector, found := db.MultiCache.Get(cacheKey); found {
			return cachedVector.([]float64), nil
		}
	}

	// 缓存未命中，计算向量
	vector, err := db.GetVectorForText(text, vectorizedType)
	if err != nil {
		return nil, err
	}

	// 更新缓存
	if db.MultiCache != nil {
		db.MultiCache.Put(cacheKey, vector)
	}

	return vector, nil
}

// GetVectorForText 将文本根据指定的向量化类型转换为向量
func (db *VectorDB) GetVectorForText(text string, vectorizedType int) ([]float64, error) {
	var vectorized DocumentVectorized
	// 注意：WordEmbeddingVectorized 可能需要预加载词向量文件路径，这里暂时硬编码或假设已加载
	// 实际应用中，这个路径应该可配置
	switch vectorizedType {
	case TfidfVectorized:
		vectorized = TFIDFVectorized() // TFIDFVectorized 内部管理其状态，每次调用可能基于已处理文档更新
	case SimpleVectorized:
		vectorized = SimpleBagOfWordsVectorized()
	case WordEmbeddingVectorized:
		// 假设 LoadWordEmbeddings 在 NewVectorDB 或其他初始化阶段被调用并存储了 embeddings
		// 或者在这里按需加载，但这效率较低。更好的方式是 VectorDB 持有 embeddings。
		// For now, let's assume a path or that embeddings are globally available/configured.
		embeddings, err := LoadWordEmbeddings("path/to/pretrained_embeddings.txt") // Placeholder path
		if err != nil {
			return nil, fmt.Errorf("failed to load word embeddings for GetVectorForText: %w", err)
		}
		vectorized = EnhancedWordEmbeddingVectorized(embeddings) // Assuming EnhancedWordEmbeddingVectorized exists
	case DefaultVectorized:
		vectorized = SimpleBagOfWordsVectorized()
	default:
		return nil, fmt.Errorf("unhandled vectorizedType: %d", vectorizedType)
	}

	if vectorized == nil {
		return nil, fmt.Errorf("vectorized function is nil for type: %d", vectorizedType)
	}

	vector, err := vectorized(text)
	if err != nil {
		return nil, fmt.Errorf("failed to vectorize text: %w", err)
	}
	return vector, nil
}

// Add 添加向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Add(id string, vector []float64) {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 跳过空向量和nil向量
	if vector == nil || len(vector) == 0 {
		logger.Warning("跳过空向量或nil向量: %s", id)
		return
	}

	// 设置向量维度（如果尚未设置）
	if db.vectorDim == 0 && len(vector) > 0 {
		db.vectorDim = len(vector)
	} else if len(vector) != db.vectorDim && db.vectorDim > 0 {
		logger.Fatal("向量维度不匹配: 期望 %d, 实际 %d", db.vectorDim, len(vector))
	}

	db.vectors[id] = vector
	// 预计算并存储归一化向量
	db.normalizedVectors[id] = acceler.NormalizeVector(vector)
	// 如果启用了 PQ 压缩，则压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := acceler.OptimizedCompressByPQ(vector, db.pqCodebook, db.numSubVectors, db.numCentroidsPerSubVector)
		if err != nil {
			logger.Error("向量 %s 压缩失败: %v。该向量将只以原始形式存储。", id, err)
			// 根据策略，可以选择是否回滚添加操作或仅记录错误
		} else {
			if db.compressedVectors == nil {
				db.compressedVectors = make(map[string]entity.CompressedVector)
			}
			db.compressedVectors[id] = compressedVec
			logger.Trace("向量 %s 已压缩并存储。", id)
		}
	}

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		logger.Info("提示: 添加新向量后，索引已失效，请重新调用 BuildIndex()。")
	}

	// 如果启用了 HNSW 索引，增量更新 HNSW 图
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		// 使用归一化后的向量（如果启用了归一化）
		vectorToAdd := vector
		if db.useNormalization {
			if normalizedVec, exists := db.normalizedVectors[id]; exists {
				vectorToAdd = normalizedVec
			}
		}

		err := db.hnsw.AddNode(id, vectorToAdd)
		if err != nil {
			logger.Warning("增量添加向量 %s 到 HNSW 图失败: %v，索引可能不一致。", id, err)
		}
	} else if db.indexed {
		db.indexed = false // 索引失效，需要重建
		logger.Info("提示: 添加新向量后，索引已失效，请重新调用 BuildIndex()。")
	}
}

// DeleteVector 从数据库中删除指定ID的向量
func (db *VectorDB) DeleteVector(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("vector with id %s not found for deletion", id)
	}
	delete(db.vectors, id)
	delete(db.normalizedVectors, id)
	delete(db.compressedVectors, id)

	// 如果使用了IVF索引 需要从相应的簇中移除该 vectorID。
	// 这部分逻辑会比较复杂，需要遍历 clusters 或维护一个反向映射。
	if db.indexed {
		for i := range db.clusters {
			newVectorIDs := make([]string, 0, len(db.clusters[i].VectorIDs))
			for _, vecID := range db.clusters[i].VectorIDs {
				if vecID != id {
					newVectorIDs = append(newVectorIDs, vecID)
				}
			}
			db.clusters[i].VectorIDs = newVectorIDs
		}
	}

	logger.Info("Vector with id %s deleted successfully.", id)
	return nil
}

// UpdateIndexIncrementally 增量更新索引
func (db *VectorDB) UpdateIndexIncrementally(id string, vector []float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 如果索引未构建，则不需要增量更新
	if !db.indexed || len(db.clusters) == 0 {
		return nil
	}

	// 找到最近的簇
	nearestClusterIndex := -1

	// 使用归一化向量进行距离计算（如果可用）
	var queryVecForDist []float64
	if db.normalizedVectors[id] != nil {
		queryVecForDist = db.normalizedVectors[id]
	} else {
		queryVecForDist = acceler.NormalizeVector(vector) // 如果没有预计算，则动态计算
	}

	//for i, cluster := range db.clusters {
	//	// 假设簇中心也是归一化的，或者在KMeans时已处理
	//	dist, err := algorithm.EuclideanDistanceSquared(queryVecForDist, cluster.Centroid) // 或者使用余弦相似度
	//	if err != nil {
	//		log.Warning("计算到簇 %d 中心的距离失败: %v", i, err)
	//		continue
	//	}
	//	if dist < minDist {
	//		minDist = dist
	//		nearestClusterIndex = i
	//	}
	//}

	// 使用自适应策略查找最近质心
	selectedStrategy := db.GetSelectStrategy()
	centroids := make([]entity.Point, len(db.clusters))
	for i, cluster := range db.clusters {
		centroids[i] = cluster.Centroid
	}

	nearestClusterIndex, _ = acceler.AdaptiveFindNearestCentroid(queryVecForDist, centroids, selectedStrategy)

	if nearestClusterIndex != -1 {
		// 从旧的簇中移除 (如果它之前在某个簇中)
		// 注意: 这需要一种方式来追踪向量当前属于哪个簇，或者遍历所有簇移除旧的ID
		// 为简化，这里假设我们只添加新的或更新的，旧的分配关系通过其他方式处理或在rebuild时修正
		// 一个更健壮的实现可能需要一个 map[string]int 来存储 vectorID -> clusterIndex 的映射

		// 将向量ID添加到最近的簇
		// 首先检查是否已存在，避免重复添加
		found := false
		for _, vecID := range db.clusters[nearestClusterIndex].VectorIDs {
			if vecID == id {
				found = true
				break
			}
		}
		if !found {
			db.clusters[nearestClusterIndex].VectorIDs = append(db.clusters[nearestClusterIndex].VectorIDs, id)
			logger.Info("向量 %s 已增量添加到簇 %d。", id, nearestClusterIndex)
		}
	} else {
		logger.Warning("未能为向量 %s 找到最近的簇进行增量更新。", id)
	}

	return nil
}

// recalculateClusterCentroid 重新计算指定簇的中心点
func (db *VectorDB) recalculateClusterCentroid(clusterIndex int) error {
	db.mu.Lock() // 注意：这里获取了写锁，如果频繁调用且簇很多，可能需要更细粒度的锁
	defer db.mu.Unlock()

	if clusterIndex < 0 || clusterIndex >= len(db.clusters) {
		return fmt.Errorf("无效的簇索引: %d", clusterIndex)
	}

	cluster := &db.clusters[clusterIndex]
	if len(cluster.VectorIDs) == 0 {
		// log.Warning("簇 %d 为空，无法重新计算中心点。可能保持旧的中心点或置为nil。", clusterIndex)
		// 根据策略，可以选择保留旧中心点，或者如果允许，将其标记为无效/删除
		return nil // 或者返回一个特定的错误/警告
	}

	if db.vectorDim == 0 {
		// 尝试从向量中推断维度，如果之前未设置
		firstVecID := cluster.VectorIDs[0]
		if vec, ok := db.vectors[firstVecID]; ok {
			db.vectorDim = len(vec)
		} else {
			return fmt.Errorf("无法获取簇 %d 中向量 %s 的数据以确定维度", clusterIndex, firstVecID)
		}
		if db.vectorDim == 0 {
			return fmt.Errorf("向量维度为0，无法计算中心点")
		}
	}
	newCentroid := make(entity.Point, db.vectorDim)
	validVectorsCount := 0
	for _, vecID := range cluster.VectorIDs {
		var vecData []float64
		var ok bool

		// 优先使用归一化向量（如果存在且配置使用）
		// 假设：如果启用了某种形式的归一化，则在计算簇中心时也应使用归一化向量
		if len(db.normalizedVectors) > 0 {
			vecData, ok = db.normalizedVectors[vecID]
		} else {
			vecData, ok = db.vectors[vecID]
		}

		if !ok {
			logger.Warning("重新计算簇 %d 中心时，向量 %s 未找到，跳过此向量。", clusterIndex, vecID)
			continue
		}
		if len(vecData) != db.vectorDim {
			logger.Warning("向量 %s 的维度 (%d) 与期望维度 (%d) 不符，跳过此向量。", vecID, len(vecData), db.vectorDim)
			continue
		}

		for i, val := range vecData {
			newCentroid[i] += val
		}
		validVectorsCount++
	}

	if validVectorsCount == 0 {
		logger.Warning("簇 %d 中没有有效的向量来计算新的中心点，保留旧中心点。", clusterIndex)
		return nil
	}

	for i := 0; i < db.vectorDim; i++ {
		newCentroid[i] /= float64(len(cluster.VectorIDs))
	}

	cluster.Centroid = newCentroid
	logger.Info("簇 %d 的中心点已重新计算。", clusterIndex)
	return nil
}

// StartClusterCentroidUpdater 启动一个定时器，定期更新所有簇的中心点
// updateInterval: 更新间隔，例如 time.Minute * 5 表示每5分钟更新一次
func (db *VectorDB) StartClusterCentroidUpdater(interval time.Duration) {
	if !db.indexed || db.numClusters <= 0 {
		logger.Info("索引未启用或簇数量未设置，不启动簇中心更新器。")
		return
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		logger.Info("簇中心定期更新器已启动，更新间隔: %s", interval.String())

		for {
			select {
			case <-ticker.C:
				logger.Info("开始定期重新计算所有簇中心...")
				db.mu.Lock() // 获取写锁以更新簇中心
				if len(db.clusters) == 0 {
					db.mu.Unlock()
					logger.Info("当前没有簇，跳过簇中心重新计算。")
					continue
				}
				for i := range db.clusters {
					if err := db.recalculateClusterCentroid(i); err != nil {
						logger.Error("重新计算簇 %d 的中心失败: %v", i, err)
					}
				}
				db.mu.Unlock()
				logger.Info("所有簇中心重新计算完成。")
			case <-db.stopCh: // 监听停止信号
				logger.Info("接收到停止信号，簇中心更新器正在关闭...")
				return // 退出goroutine
			}
		}
	}()
}

// Get 获取向量
func (db *VectorDB) Get(id string) ([]float64, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	vec, exists := db.vectors[id]
	return vec, exists
}

// Update 更新向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Update(id string, vector []float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("未找到ID为 '%s' 的向量", id)
	}

	// 检查向量维度
	if len(vector) != db.vectorDim && db.vectorDim > 0 {
		return fmt.Errorf("向量维度不匹配: 期望 %d, 实际 %d", db.vectorDim, len(vector))
	}

	db.vectors[id] = vector
	// 更新归一化向量
	db.normalizedVectors[id] = acceler.NormalizeVector(vector)
	// 如果启用了 PQ 压缩，则更新压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := acceler.OptimizedCompressByPQ(vector, db.pqCodebook, db.numSubVectors, db.numCentroidsPerSubVector)
		if err != nil {
			logger.Error("为向量 %s 更新时压缩向量失败: %v", id, err)
			// 即使压缩失败，原始向量也已更新
			delete(db.compressedVectors, id) // 删除旧地压缩向量，因为它不再有效
		} else {
			db.compressedVectors[id] = compressedVec
		}
	}

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		logger.Info("提示: 更新向量后，索引已失效，请重新调用 BuildIndex()。")
	}
	return nil
}

// Delete 删除向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Delete(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("未找到ID为 '%s' 的向量", id)
	}
	delete(db.vectors, id)
	delete(db.normalizedVectors, id)
	delete(db.compressedVectors, id) // 删除压缩向量

	// 从倒排索引中删除
	db.invertedMu.Lock()
	for word, ids := range db.invertedIndex {
		newIDs := make([]string, 0, len(ids))
		for _, existingID := range ids {
			if existingID != id {
				newIDs = append(newIDs, existingID)
			}
		}
		if len(newIDs) == 0 {
			delete(db.invertedIndex, word)
		} else {
			db.invertedIndex[word] = newIDs
		}
	}
	db.invertedMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		logger.Info("提示: 删除向量后，索引已失效，请重新调用 BuildIndex()。")
	}

	// 如果启用了 HNSW 索引，从 HNSW 图中删除节点
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		err := db.hnsw.DeleteNode(id)
		if err != nil {
			logger.Warning("从 HNSW 图中删除向量 %s 失败: %v，索引可能不一致。", id, err)
		}
	} else if db.indexed {
		db.indexed = false // 索引失效，需要重建
		logger.Info("提示: 删除向量后，索引已失效，请重新调用 BuildIndex()。")
	}

	return nil
}

// MultiLevelIndex 多级索引结构
type MultiLevelIndex struct {
	// 一级索引：簇中心
	Clusters []Cluster

	// 二级索引：每个簇内部的KD树或其他数据结构
	SubIndices []interface{} // 可以是KDTree或其他索引结构

	// 索引元数据
	NumClusters int
	Indexed     bool
	BuildTime   time.Time
}

// BuildMultiLevelIndex 在BuildIndex方法中构建多级索引
func (db *VectorDB) BuildMultiLevelIndex(maxIterations int, tolerance float64) error {
	// 检查参数有效性
	if db.numClusters <= 0 {
		return fmt.Errorf("未配置有效的簇数量 (numClusters: %d)，无法构建索引", db.numClusters)
	}

	if len(db.vectors) < db.numClusters {
		db.indexed = false
		return fmt.Errorf("向量数量 (%d) 少于簇数量 (%d)，无法构建有效索引", len(db.vectors), db.numClusters)
	}

	fmt.Println("开始构建多级索引...")
	startTime := time.Now()

	// 1. 收集所有向量及其ID
	var allVectorsData []entity.Point
	var vectorIDs []string // 保持与allVectorsData顺序一致的ID
	for id, vec := range db.vectors {
		allVectorsData = append(allVectorsData, vec)
		vectorIDs = append(vectorIDs, id)
	}

	// 2. 第一级：构建IVF索引（调用KMeans算法）
	centroids, assignments, err := algorithm.KMeans(allVectorsData, db.numClusters, maxIterations, tolerance)
	if err != nil {
		return fmt.Errorf("KMeans聚类失败: %w", err)
	}

	// 3. 根据KMeans结果填充db.clusters
	db.clusters = make([]Cluster, db.numClusters)
	for i := 0; i < db.numClusters; i++ {
		db.clusters[i] = Cluster{
			Centroid:  centroids[i],
			VectorIDs: make([]string, 0),
		}
	}

	for i, clusterIndex := range assignments {
		if clusterIndex >= 0 && clusterIndex < db.numClusters { // 确保索引有效
			db.clusters[clusterIndex].VectorIDs = append(db.clusters[clusterIndex].VectorIDs, vectorIDs[i])
		} else {
			fmt.Printf("警告: 向量 %s 被分配到无效的簇索引 %d\n", vectorIDs[i], clusterIndex)
		}
	}

	// 4. 第二级：为每个簇构建KD树子索引
	multiIndex := &MultiLevelIndex{
		Clusters:    db.clusters,                         // 注意：这里可能需要深拷贝或调整，取决于 MultiLevelIndex 的设计
		SubIndices:  make([]interface{}, db.numClusters), // 假设 subIndices 在 goroutine 中填充
		NumClusters: db.numClusters,
		Indexed:     true,
		BuildTime:   time.Now(),
	}

	// 并行构建每个簇的KD树
	var wg sync.WaitGroup
	for i := range db.clusters {
		wg.Add(1)
		go func(clusterIdx int) {
			defer wg.Done()

			// 跳过空簇
			if len(db.clusters[clusterIdx].VectorIDs) == 0 {
				return
			}

			// 创建KD树
			kdTree := tree.NewKDTree(db.vectorDim)

			// 将簇内所有向量插入KD树
			for _, id := range db.clusters[clusterIdx].VectorIDs {
				vec, exists := db.vectors[id]
				if exists {
					kdTree.Insert(vec, id)
				}
			}

			// 保存KD树到多级索引
			multiIndex.SubIndices[clusterIdx] = kdTree
		}(i)
	}

	// 等待所有KD树构建完成
	wg.Wait()

	db.multiIndex = multiIndex // 保存构建好的多级索引
	db.config.UseMultiLevelIndex = true
	// 更新索引状态
	db.indexed = true

	// 更新性能统计
	db.statsMu.Lock()
	db.stats.IndexBuildTime = time.Since(startTime)
	db.stats.LastReindexTime = time.Now()

	// 估算内存使用
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	db.stats.MemoryUsage = memStats.Alloc
	db.statsMu.Unlock()

	fmt.Printf("多级索引构建完成，共 %d 个簇，耗时 %v\n", db.numClusters, time.Since(startTime))
	return nil
}

// BuildIndex 使用K-Means算法为数据库中的向量构建索引。
// maxIterations: K-Means的最大迭代次数。
// tolerance: K-Means的收敛容忍度。
func (db *VectorDB) BuildIndex(maxIterations int, tolerance float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 检查是否启用了多级索引
	if db.config.UseMultiLevelIndex {
		// 使用多级索引构建
		return db.BuildMultiLevelIndex(maxIterations, tolerance)
	}

	// 以下是原有的单级索引构建逻辑
	if db.numClusters <= 0 {
		return fmt.Errorf("未配置有效的簇数量 (numClusters: %d)，无法构建索引", db.numClusters)
	}

	if len(db.vectors) < db.numClusters {
		db.indexed = false
		return fmt.Errorf("向量数量 (%d) 少于簇数量 (%d)，无法构建有效索引", len(db.vectors), db.numClusters)
	}

	logger.Info("开始构建索引...")
	// 1. 收集所有向量及其ID
	var allVectorsData []entity.Point
	var vectorIDs []string // 保持与allVectorsData顺序一致的ID
	for id, vec := range db.vectors {
		allVectorsData = append(allVectorsData, vec)
		vectorIDs = append(vectorIDs, id)
	}

	// 2. 调用KMeans算法
	centroids, assignments, err := algorithm.KMeans(allVectorsData, db.numClusters, maxIterations, tolerance)
	if err != nil {
		return fmt.Errorf("KMeans聚类失败: %w", err)
	}

	// 3. 根据KMeans结果填充db.clusters
	db.clusters = make([]Cluster, db.numClusters)
	for i := 0; i < db.numClusters; i++ {
		db.clusters[i] = Cluster{
			Centroid:  centroids[i],
			VectorIDs: make([]string, 0),
		}
	}

	for i, clusterIndex := range assignments {
		if clusterIndex >= 0 && clusterIndex < db.numClusters { // 确保索引有效
			db.clusters[clusterIndex].VectorIDs = append(db.clusters[clusterIndex].VectorIDs, vectorIDs[i])
		} else {
			logger.Warning("警告: 向量 %s 被分配到无效的簇索引 %d\n", vectorIDs[i], clusterIndex)
		}
	}

	db.indexed = true

	logger.Info("索引构建完成，共 %d 个簇。\n", db.numClusters)
	return nil
}

// 自适应 nprobe 设置
func (db *VectorDB) getAdaptiveNprobe() int {
	vectorCount := len(db.vectors)

	// 根据数据规模动态调整 nprobe
	if vectorCount > 1000000 {
		return db.numClusters / 10 // 大数据集，搜索更多簇
	} else if vectorCount > 100000 {
		return db.numClusters / 20 // 中等数据集
	} else {
		return max(1, db.numClusters/50) // 小数据集
	}
}

// estimateDataSize 估算序列化数据大小
func (db *VectorDB) estimateDataSize() int64 {
	// 粗略估算：向量数量 * 向量维度 * 8字节 + 其他数据
	vectorSize := int64(len(db.vectors)) * int64(db.vectorDim) * 8
	otherDataSize := int64(1024 * 1024) // 1MB 用于其他数据
	return vectorSize + otherDataSize
}

// initializeEmptyDB 初始化空数据库
func (db *VectorDB) initializeEmptyDB() {
	db.vectors = make(map[string][]float64)
	db.clusters = make([]Cluster, 0)
	db.indexed = false
	db.invertedIndex = make(map[string][]string)
	db.normalizedVectors = make(map[string][]float64)
	db.compressedVectors = make(map[string]entity.CompressedVector)
	db.pqCodebook = nil
}

// restoreDataFromStruct 从结构体恢复数据
func (db *VectorDB) restoreDataFromStruct(data struct {
	Vectors                  map[string][]float64
	Clusters                 []Cluster
	NumClusters              int
	Indexed                  bool
	InvertedIndex            map[string][]string
	VectorDim                int
	VectorizedType           int
	NormalizedVectors        map[string][]float64
	CompressedVectors        map[string]entity.CompressedVector
	UseCompression           bool
	PqCodebookFilePath       string
	NumSubVectors            int
	NumCentroidsPerSubVector int
	UsePQCompression         bool
	UseHNSWIndex             bool
	MaxConnections           int
	EfConstruction           float64
	EfSearch                 float64
}) {
	db.vectors = data.Vectors
	db.clusters = data.Clusters
	db.numClusters = data.NumClusters
	db.indexed = data.Indexed
	db.invertedIndex = data.InvertedIndex
	db.vectorDim = data.VectorDim
	db.vectorizedType = data.VectorizedType
	db.normalizedVectors = data.NormalizedVectors
	db.compressedVectors = data.CompressedVectors
	db.useCompression = data.UseCompression
	db.pqCodebookFilePath = data.PqCodebookFilePath
	db.numSubVectors = data.NumSubVectors
	db.numCentroidsPerSubVector = data.NumCentroidsPerSubVector
	db.usePQCompression = data.UsePQCompression
	db.useHNSWIndex = data.UseHNSWIndex
	db.maxConnections = data.MaxConnections
	db.efConstruction = data.EfConstruction
	db.efSearch = data.EfSearch

	// 确保 map 不为 nil
	if db.vectors == nil {
		db.vectors = make(map[string][]float64)
	}
	if db.invertedIndex == nil {
		db.invertedIndex = make(map[string][]string)
	}
	if db.normalizedVectors == nil {
		db.normalizedVectors = make(map[string][]float64)
	}
	if db.compressedVectors == nil {
		db.compressedVectors = make(map[string]entity.CompressedVector)
	}
}

// CalculateApproximateDistancePQ 以下是一个基于 PQ 的近似距离计算的简化示例，需要集成到 FindNearest 或新的搜索函数中
// CalculateApproximateDistancePQ 计算查询向量与数据库中压缩向量的近似距离 (ADC)
func (db *VectorDB) CalculateApproximateDistancePQ(queryVector []float64, compressedDBVector entity.CompressedVector) (float64, error) {
	if !db.usePQCompression || db.pqCodebook == nil {
		return 0, fmt.Errorf("PQ 压缩未启用或码本未设置")
	}
	if len(queryVector) != db.vectorDim {
		return 0, fmt.Errorf("查询向量维度 %d 与数据库向量维度 %d 不匹配", len(queryVector), db.vectorDim)
	}

	// 调用 util 中的 PQ 近似距离计算函数
	dist, err := acceler.ApproximateDistanceADC(queryVector, compressedDBVector, db.pqCodebook, db.numSubVectors)
	if err != nil {
		return 0, fmt.Errorf("计算 PQ 近似距离失败: %w", err)
	}
	return dist, nil
}

func (db *VectorDB) CalculateAvgQueryTime(startTime time.Time) time.Duration {
	avgTimeMicroSeconds := db.stats.AvgQueryTime.Microseconds()
	startTimeMicroSeconds := time.Since(startTime).Microseconds()
	return time.Duration(((avgTimeMicroSeconds * (db.stats.TotalQueries - 1)) + startTimeMicroSeconds) / db.stats.TotalQueries)
}

// UpdateHNSWIndexIncrementally 增量更新 HNSW 索引
func (db *VectorDB) UpdateHNSWIndexIncrementally(id string, vector []float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if !db.useHNSWIndex || db.hnsw == nil {
		return fmt.Errorf("HNSW 索引未启用或未初始化")
	}

	// 检查节点是否已存在
	if _, exists := db.hnsw.GetNode(id); exists {
		// 如果节点已存在，先删除
		if err := db.hnsw.DeleteNode(id); err != nil {
			return fmt.Errorf("删除现有节点失败: %w", err)
		}
	}

	// 添加新节点
	if err := db.hnsw.AddNode(id, vector); err != nil {
		return fmt.Errorf("添加节点到 HNSW 图失败: %w", err)
	}

	return nil
}

// OptimizeHNSWParameters 根据数据集大小自适应调整 HNSW 参数
func (db *VectorDB) OptimizeHNSWParameters() {
	db.mu.Lock()
	defer db.mu.Unlock()

	dataSize := len(db.vectors)

	// 根据数据集大小调整参数
	if dataSize < 1000 {
		db.maxConnections = 16
		db.efConstruction = 100
		db.efSearch = 50
	} else if dataSize < 10000 {
		db.maxConnections = 32
		db.efConstruction = 200 // 增加构建质量
		db.efSearch = 100
	} else if dataSize < 100000 {
		db.maxConnections = 64
		db.efConstruction = 400
		db.efSearch = 200
	} else if dataSize < 1000000 {
		db.maxConnections = 96
		db.efConstruction = 600
		db.efSearch = 300
	} else {
		db.maxConnections = 128
		db.efConstruction = 800 // 大数据集需要更高质量的图
		db.efSearch = 400
	}

	// 如果 HNSW 索引已启用，更新参数
	if db.useHNSWIndex && db.hnsw != nil {
		// 注意：这里只能更新 efSearch，其他参数需要重建索引
		// 可以考虑添加一个标志，在下次重建索引时使用新参数
		db.hnsw.EfSearch = db.efSearch
	}
}

// BatchNormalizeVectors 批量归一化向量
func (db *VectorDB) BatchNormalizeVectors() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if !db.useNormalization {
		return fmt.Errorf("归一化未启用")
	}

	// 并行归一化
	numWorkers := runtime.NumCPU()
	vectorChan := make(chan struct {
		id  string
		vec []float64
	}, len(db.vectors))
	resultChan := make(chan struct {
		id         string
		normalized []float64
	}, len(db.vectors))

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range vectorChan {
				normalized := acceler.NormalizeVector(item.vec)
				resultChan <- struct {
					id         string
					normalized []float64
				}{
					id: item.id, normalized: normalized,
				}
			}
		}()
	}

	// 发送任务
	go func() {
		for id, vec := range db.vectors {
			vectorChan <- struct {
				id  string
				vec []float64
			}{id: id, vec: vec}
		}
		close(vectorChan)
	}()

	// 收集结果
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 更新归一化向量
	for result := range resultChan {
		db.normalizedVectors[result.id] = result.normalized
	}

	logger.Info("批量归一化完成，处理了 %d 个向量", len(db.vectors))
	return nil
}

// EnableGPUAcceleration 为 VectorDB 启用 GPU 加速
func (db *VectorDB) EnableGPUAcceleration(gpuID int, indexType string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 检查硬件管理器是否初始化
	if db.hardwareManager == nil {
		return fmt.Errorf("硬件管理器未初始化")
	}

	// 创建 GPU 加速器
	gpuAccel := acceler.NewFAISSAccelerator(gpuID, indexType)

	// 初始化 GPU 加速器
	err := gpuAccel.Initialize()
	if err != nil {
		return fmt.Errorf("初始化 GPU 加速器失败: %w", err)
	}

	// 注册到硬件管理器
	if err := db.hardwareManager.RegisterGPUAccelerator(gpuAccel); err != nil {
		return fmt.Errorf("注册 GPU 加速器失败: %w", err)
	}

	logger.Info("GPU 加速已启用，GPU ID: %d, 索引类型: %s", gpuID, indexType)
	return nil
}

// TwoStageSearch 两阶段搜索实现
func (db *VectorDB) TwoStageSearch(query []float64, config TwoStageSearchConfig) ([]entity.Result, error) {
	startTime := time.Now()

	// 第一阶段：粗筛
	coarseCandidates, err := db.coarseSearch(query, config.CoarseK, config.CoarseNprobe)
	if err != nil {
		return nil, fmt.Errorf("粗筛阶段失败: %w", err)
	}

	logger.Trace("粗筛阶段完成，获得 %d 个候选", len(coarseCandidates))

	// 第二阶段：精排
	finalResults, err := db.fineRanking(query, coarseCandidates, config)
	if err != nil {
		return nil, fmt.Errorf("精排阶段失败: %w", err)
	}

	// 更新统计信息
	db.statsMu.Lock()
	db.stats.TotalQueries++
	queryTime := time.Since(startTime)
	db.stats.AvgQueryTime = time.Duration((db.stats.AvgQueryTime.Nanoseconds()*9 + queryTime.Nanoseconds()) / 10)
	db.statsMu.Unlock()

	logger.Trace("两阶段搜索完成，耗时 %v", queryTime)
	return finalResults, nil
}

// coarseSearch 粗筛阶段：快速筛选候选集
func (db *VectorDB) coarseSearch(query []float64, k int, nprobe int) ([]string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.indexed {
		// 如果没有索引，使用简化的距离计算
		return db.approximateSearch(query, k)
	}

	// 使用 IVF 索引进行粗筛
	normalizedQuery := acceler.NormalizeVector(query)

	// 找到最近的 nprobe 个簇
	nearestClusters := make([]int, 0, nprobe)
	clusterDists := make([]float64, len(db.clusters))

	// 获取最佳计算策略
	selectedStrategy := db.GetSelectStrategy()

	for i, cluster := range db.clusters {
		dist, _ := acceler.AdaptiveEuclideanDistanceSquared(normalizedQuery, cluster.Centroid, selectedStrategy)
		clusterDists[i] = dist
	}

	// 选择距离最小的 nprobe 个簇
	for i := 0; i < nprobe && i < len(db.clusters); i++ {
		minIdx := 0
		minDist := clusterDists[0]
		for j, dist := range clusterDists {
			if dist < minDist {
				minDist = dist
				minIdx = j
			}
		}
		nearestClusters = append(nearestClusters, minIdx)
		clusterDists[minIdx] = math.MaxFloat64 // 标记已选择
	}

	// 收集候选向量 ID
	candidates := make([]string, 0, k*2)
	for _, clusterIdx := range nearestClusters {
		candidates = append(candidates, db.clusters[clusterIdx].VectorIDs...)
		if len(candidates) >= k*2 {
			break
		}
	}

	// 限制候选数量
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates, nil
}

func (db *VectorDB) appendResults(query []float64, resultStrings []string, results *[]entity.Result, config TwoStageSearchConfig) {
	for _, id := range resultStrings {
		vec, exists := db.vectors[id]
		if !exists {
			continue
		}
		similarity := acceler.OptimizedCosineSimilarity(query, vec)
		if similarity < config.Threshold {
			continue
		}

		*results = append(*results, entity.Result{
			Id:         id,
			Similarity: similarity,
		})
	}
}

// fineRanking 精排阶段：精确计算相似度并排序
func (db *VectorDB) fineRanking(query []float64, candidates []string, config TwoStageSearchConfig) ([]entity.Result, error) {
	if len(candidates) == 0 {
		return []entity.Result{}, nil
	}

	results := make([]entity.Result, 0, len(candidates))

	if config.UseGPU {
		// 使用 GPU 加速精排
		gpuResults, err := db.gpuFineRanking(query, candidates, config.FineK)
		if err != nil {
			// GPU 失败时回退到 CPU
			cpuResults, cpuErr := db.cpuFineRanking(query, candidates, config.FineK, acceler.StrategyStandard)
			if cpuErr != nil {
				return nil, cpuErr
			}

			return cpuResults, nil
		}

		// 将 GPU 结果转换为 entity.Result 格式
		//for _, id := range gpuResults {
		//	if vec, exists := db.vectors[id]; exists {
		//		similarity := algorithm.OptimizedCosineSimilarity(query, vec)
		//		if similarity >= config.Threshold {
		//			results = append(results, entity.Result{
		//				Id:         id,
		//				Similarity: similarity,
		//			})
		//		}
		//	}
		//}

		return gpuResults, nil
	}

	// CPU 并行精排
	numWorkers := runtime.NumCPU()
	candidateChan := make(chan string, len(candidates))
	resultChan := make(chan entity.Result, len(candidates))

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for id := range candidateChan {
				vec, exists := db.vectors[id]
				if !exists {
					continue
				}

				// 使用优化的相似度计算
				similarity := acceler.OptimizedCosineSimilarity(query, vec)

				// 应用阈值过滤
				if similarity >= config.Threshold {
					resultChan <- entity.Result{
						Id:         id,
						Similarity: similarity,
					}
				}
			}
		}()
	}

	// 发送候选
	go func() {
		for _, id := range candidates {
			candidateChan <- id
		}
		close(candidateChan)
	}()

	// 收集结果
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for result := range resultChan {
		results = append(results, result)
	}

	// 按相似度降序排序
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// 限制返回数量
	if len(results) > config.FineK {
		results = results[:config.FineK]
	}

	return results, nil
}

// approximateSearch 使用简化的距离计算进行快速搜索
func (db *VectorDB) approximateSearch(query []float64, k int) ([]string, error) {
	if len(db.vectors) == 0 {
		return []string{}, nil
	}

	// 归一化查询向量
	normalizedQuery := acceler.NormalizeVector(query)

	// 使用简化的距离计算（只计算前几个维度的距离作为近似）
	approxDim := int(math.Min(float64(len(normalizedQuery)), 32)) // 只使用前32个维度进行近似计算
	if approxDim < 8 {
		approxDim = len(normalizedQuery) // 如果维度太少，使用全部维度
	}

	type candidate struct {
		id   string
		dist float64
	}

	candidates := make([]candidate, 0, len(db.vectors))

	// 并行计算简化距离
	numWorkers := runtime.NumCPU()
	vectorChan := make(chan struct {
		id  string
		vec []float64
	}, len(db.vectors))
	resultChan := make(chan candidate, len(db.vectors))

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range vectorChan {
				// 使用简化的欧氏距离计算（只计算前几个维度）
				dist := 0.0
				for j := 0; j < approxDim; j++ {
					if j < len(item.vec) {
						diff := normalizedQuery[j] - item.vec[j]
						dist += diff * diff
					}
				}

				resultChan <- candidate{
					id:   item.id,
					dist: dist,
				}
			}
		}()
	}

	// 发送向量数据
	go func() {
		for id, vec := range db.vectors {
			vectorChan <- struct {
				id  string
				vec []float64
			}{
				id:  id,
				vec: vec,
			}
		}
		close(vectorChan)
	}()

	// 收集结果
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for candidate := range resultChan {
		candidates = append(candidates, candidate)
	}

	// 按距离升序排序（距离越小越相似）
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	// 返回前k个结果的ID
	resultCount := int(math.Min(float64(k), float64(len(candidates))))
	results := make([]string, resultCount)
	for i := 0; i < resultCount; i++ {
		results[i] = candidates[i].id
	}

	return results, nil
}

// FindNearest 查找最近邻（更新为使用自适应计算）
func (db *VectorDB) FindNearest(query []float64, k int, nprobe int) ([]entity.Result, error) {
	return db.AdaptiveFindNearest(query, k, nprobe)
}

//// FindNearest 优化后的FindNearest方法
//func (db *VectorDB) FindNearest_1(query []float64, k int, nprobe int) ([]string, error) {
//	startTime := time.Now()
//	// 更新查询计数
//	db.statsMu.Lock()
//	db.stats.TotalQueries++
//	db.statsMu.Unlock()
//	// 检查缓存
//	cacheKey := algorithm.GenerateCacheKey(query, k, nprobe, 0)
//	db.cacheMu.RLock()
//	cached, found := db.queryCache[cacheKey]
//	if found && (time.Now().Unix()-cached.timestamp) < db.cacheTTL {
//		db.cacheMu.RUnlock()
//		db.statsMu.Lock()
//		db.stats.CacheHits++
//		db.stats.AvgQueryTime = db.CalculateAvgQueryTime(startTime)
//		db.statsMu.Unlock()
//		return cached.results, nil
//	}
//	db.cacheMu.RUnlock()
//
//	db.mu.RLock()
//	defer db.mu.RUnlock()
//
//	if k <= 0 {
//		return nil, fmt.Errorf("k 必须是正整数")
//	}
//
//	if len(db.vectors) == 0 {
//		return []string{}, nil
//	}
//	var results []entity.Result
//	var err error
//	// 如果启用了 HNSW 索引，使用 HNSW 搜索
//	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
//		log.Trace("使用 HNSW 索引进行搜索。")
//
//		// 预处理查询向量
//		normalizedQuery := algorithm.NormalizeVector(query)
//
//		// 使用 HNSW 搜索
//		results, err := db.hnsw.Search(normalizedQuery, k)
//		if err != nil {
//			return nil, fmt.Errorf("HNSW 搜索失败: %w", err)
//		}
//
//		// 提取 ID
//		ids := make([]string, len(results))
//		for i, result := range results {
//			ids[i] = result.Id
//		}
//
//		return ids, nil
//	}
//	if !db.indexed {
//		// 如果未索引，执行暴力搜索
//		log.Warning("VectorDB is not indexed. Performing brute-force search.")
//		results, err = db.bruteForceSearch(query, k)
//		if err != nil {
//			return nil, err
//		}
//	} else if db.config.UseMultiLevelIndex && db.multiIndex != nil && db.multiIndex.Indexed {
//		results, err = db.multiIndexSearch(query, k, nprobe)
//		if err != nil {
//			return nil, err
//		}
//	} else {
//		vectorCount := len(db.GetVectors())
//		options := SearchOptions{
//			Nprobe:        nprobe,
//			NumHashTables: 4 + vectorCount/10000, // 根据数据规模调整哈希表数量
//			UseANN:        true,
//		}
//
//		results, err = db.ivfSearch(query, k, options.Nprobe)
//		if err != nil {
//			return nil, err
//		}
//	}
//	finalResults := make([]string, len(results))
//	for i := len(results) - 1; i >= 0; i-- {
//		finalResults[len(results)-1-i] = results[i].Id
//	}
//
//	// 更新平均查询时间统计
//	db.statsMu.Lock()
//	// 使用指数移动平均更新平均查询时间
//	// queryTime := time.Since(startTime)
//	// alpha := 0.1
//	// db.stats.AvgQueryTime = time.Duration(float64(db.stats.AvgQueryTime)*(1-alpha) + float64(queryTime)*alpha)
//	db.stats.AvgQueryTime = db.CalculateAvgQueryTime(startTime)
//	db.statsMu.Unlock()
//
//	return finalResults, nil
//}

func (db *VectorDB) GetOptimalStrategy(query []float64) acceler.ComputeStrategy {
	dataSize := len(db.vectors)
	vectorDim := len(query)
	optimalStrategy := db.strategyComputeSelector.SelectOptimalStrategy(dataSize, vectorDim)

	return optimalStrategy
}

// FindNearestWithScores 查找最近邻并返回分数（更新为使用自适应计算）
func (db *VectorDB) FindNearestWithScores(query []float64, k int, nprobe int) ([]entity.Result, error) {
	startTime := time.Now()
	defer func() {
		queryTime := time.Since(startTime)
		db.statsMu.Lock()
		db.stats.TotalQueries++
		if db.stats.AvgQueryTime == 0 {
			db.stats.AvgQueryTime = queryTime
		} else {
			alpha := 0.1
			db.stats.AvgQueryTime = time.Duration(float64(db.stats.AvgQueryTime)*(1-alpha) + float64(queryTime)*alpha)
		}
		db.statsMu.Unlock()
	}()

	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("数据库为空，无法进行搜索")
	}

	// 选择最优计算策略
	selectStrategy := db.GetOptimalStrategy(query)

	// 如果启用了HNSW索引，优先使用
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		// 设置自适应距离函数
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := acceler.AdaptiveCosineSimilarity(a, b, selectStrategy)
			return 1.0 - sim, nil
		})

		normalizedQuery := acceler.NormalizeVector(query)
		results, err := db.hnsw.Search(normalizedQuery, k)
		return results, err
	}

	// 使用IVF索引进行自适应搜索
	if !db.indexed {
		return nil, fmt.Errorf("数据库尚未建立索引")
	}

	// IVF搜索逻辑：粗排 + 精排
	return db.ivfSearchWithScores(query, k, nprobe, selectStrategy)
}

// ivfSearchWithScores IVF搜索返回带分数的结果
func (db *VectorDB) ivfSearchWithScores(query []float64, k int, nprobe int, strategy acceler.ComputeStrategy) ([]entity.Result, error) {
	// 粗排：找到最近的nprobe个簇
	candidateClusters := make([]int, 0, nprobe)
	clusterDistances := make([]float64, len(db.clusters))

	for i, cluster := range db.clusters {
		// 使用自适应距离计算
		switch strategy {
		case acceler.StrategyAVX512, acceler.StrategyAVX2:
			sim := acceler.AdaptiveCosineSimilarity(query, cluster.Centroid, strategy)
			clusterDistances[i] = 1.0 - sim
		default:
			sim := acceler.CosineSimilarity(query, cluster.Centroid)
			clusterDistances[i] = 1.0 - sim
		}
	}

	// 选择距离最近的nprobe个簇
	type clusterDist struct {
		index    int
		distance float64
	}

	clusterList := make([]clusterDist, len(db.clusters))
	for i, dist := range clusterDistances {
		clusterList[i] = clusterDist{index: i, distance: dist}
	}

	sort.Slice(clusterList, func(i, j int) bool {
		return clusterList[i].distance < clusterList[j].distance
	})

	for i := 0; i < nprobe && i < len(clusterList); i++ {
		candidateClusters = append(candidateClusters, clusterList[i].index)
	}

	// 精排：在候选簇中搜索最近邻并返回带分数的结果
	candidateVectors := make([]string, 0)
	for _, clusterIdx := range candidateClusters {
		candidateVectors = append(candidateVectors, db.clusters[clusterIdx].VectorIDs...)
	}

	return db.fineRankingWithScores(query, candidateVectors, k, strategy)
}

// fineRankingWithScores 精排并返回带分数的结果
func (db *VectorDB) fineRankingWithScores(query []float64, candidates []string, k int, strategy acceler.ComputeStrategy) ([]entity.Result, error) {
	if len(candidates) == 0 {
		return []entity.Result{}, nil
	}

	// 记录开始时间，用于性能分析
	startTime := time.Now()

	// 获取GPU加速器
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()

	// 检查是否应该使用GPU加速
	if db.shouldUseGPUBatchSearch(1, len(candidates)) && len(candidates) > 100 {
		// 准备候选向量数组
		candidateVectors := make([][]float64, 0, len(candidates))
		idMapping := make([]string, 0, len(candidates))

		// 构建向量数组和ID映射
		for _, candidateID := range candidates {
			if vec, exists := db.vectors[candidateID]; exists {
				candidateVectors = append(candidateVectors, vec)
				idMapping = append(idMapping, candidateID)
			}
		}

		// 使用GPU加速器进行搜索
		if gpuAccelerator != nil && len(candidateVectors) > 0 {
			logger.Debug("使用GPU加速单向量搜索，候选向量数量: %d", len(candidateVectors))

			// 将单个查询向量包装为批量查询
			queries := [][]float64{query}

			// 调用GPU加速器的BatchSearch方法
			gpuResults, err := gpuAccelerator.BatchSearch(queries, candidateVectors, k)

			// 计算耗时
			elapsed := time.Since(startTime)

			// 错误处理
			if err != nil {
				logger.Warning("GPU单向量搜索失败(耗时: %v)，回退到CPU搜索: %v", elapsed, err)

				// 更新统计信息
				db.statsMu.Lock()
				db.stats.GPUErrors++
				db.statsMu.Unlock()
			} else if len(gpuResults) > 0 {
				// 转换GPU结果为entity.Result格式
				results := make([]entity.Result, min(k, len(gpuResults[0])))
				for j, gpuResult := range gpuResults[0] {
					// 将GPU返回的数字ID转换为实际的向量ID
					gpuID := gpuResult.ID
					if idx, err := strconv.Atoi(gpuID); err == nil && idx < len(idMapping) {
						results[j] = entity.Result{
							Id:         idMapping[idx],
							Similarity: gpuResult.Similarity,
						}
					} else {
						// 如果ID转换失败，使用原始ID
						logger.Warning("GPU结果ID转换失败: %s", gpuID)
						results[j] = entity.Result{
							Id:         gpuID,
							Similarity: gpuResult.Similarity,
						}
					}
				}

				// 更新统计信息
				db.statsMu.Lock()
				db.stats.GPUSearches++
				db.stats.TotalGPUSearchTime += elapsed.Milliseconds()
				db.statsMu.Unlock()

				logger.Debug("GPU单向量搜索完成，耗时: %v，结果数量: %d", elapsed, len(results))
				return results, nil
			}
		}
	}

	// 回退到CPU搜索
	results := make([]entity.Result, 0, len(candidates))
	resultsChan := make(chan entity.Result, len(candidates))

	// 并行计算相似度
	numWorkers := runtime.NumCPU()
	workChan := make(chan string, len(candidates))
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for candidateID := range workChan {
				if vec, exists := db.vectors[candidateID]; exists {
					// 使用自适应相似度计算
					var sim float64
					switch strategy {
					case acceler.StrategyAVX512, acceler.StrategyAVX2:
						sim = acceler.AdaptiveCosineSimilarity(query, vec, strategy)
					default:
						sim = acceler.CosineSimilarity(query, vec)
					}

					resultsChan <- entity.Result{
						Id:         candidateID,
						Similarity: sim,
					}
				}
			}
		}()
	}

	// 发送任务
	for _, candidateID := range candidates {
		workChan <- candidateID
	}
	close(workChan)

	wg.Wait()
	close(resultsChan)

	// 收集结果
	for result := range resultsChan {
		results = append(results, result)
	}

	// 排序并返回top-k
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if k > len(results) {
		k = len(results)
	}

	return results[:k], nil
}

// Findnearestwithscores1 查找最近的k个向量，并返回它们的ID和相似度分数
func (db *VectorDB) Findnearestwithscores1(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	startTime := time.Now()
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}
	if len(db.vectors) == 0 {
		return []entity.Result{}, nil
	}
	normalizedQuery := acceler.NormalizeVector(query)
	var results []entity.Result
	if db.indexed && len(db.clusters) > 0 && db.numClusters > 0 {
		if nprobe <= 0 {
			nprobe = 1
		}
		if nprobe > db.numClusters {
			nprobe = db.numClusters
		}
		centroidHeap := make(entity.CentroidHeap, 0, db.numClusters)
		for i, cluster := range db.clusters {
			distSq, err := algorithm.EuclideanDistanceSquared(normalizedQuery, cluster.Centroid)
			if err != nil {
				continue
			}
			centroidHeap = append(centroidHeap, entity.CentroidDist{Index: i, Distance: distSq})
		}
		heap.Init(&centroidHeap)
		var nearestClusters []int
		for i := 0; i < nprobe && len(centroidHeap) > 0; i++ {
			nearestClusters = append(nearestClusters, heap.Pop(&centroidHeap).(entity.CentroidDist).Index)
		}
		resultHeap := make(entity.ResultHeap, 0, k)
		heap.Init(&resultHeap)
		for _, clusterIndex := range nearestClusters {
			selectedCluster := db.clusters[clusterIndex]
			batchSize := 100
			for i := 0; i < len(selectedCluster.VectorIDs); i += batchSize {
				end := i + batchSize
				if end > len(selectedCluster.VectorIDs) {
					end = len(selectedCluster.VectorIDs)
				}
				var wg sync.WaitGroup
				resultChan := make(chan entity.Result, end-i)
				for j := i; j < end; j++ {
					wg.Add(1)
					go func(vecID string) {
						defer wg.Done()
						vec, exists := db.vectors[vecID]
						if !exists {
							return
						}
						sim := acceler.CosineSimilarity(normalizedQuery, vec)
						resultChan <- entity.Result{Id: vecID, Similarity: sim}
					}(selectedCluster.VectorIDs[j])
				}
				go func() {
					wg.Wait()
					close(resultChan)
				}()
				for res := range resultChan {
					if len(resultHeap) < k {
						heap.Push(&resultHeap, res)
					} else if res.Similarity > resultHeap[0].Similarity {
						heap.Pop(&resultHeap)
						heap.Push(&resultHeap, res)
					}
				}
			}
		}
		results = make([]entity.Result, len(resultHeap))
		for i := len(resultHeap) - 1; i >= 0; i-- {
			results[i] = heap.Pop(&resultHeap).(entity.Result)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Similarity > results[j].Similarity
		})
	} else {
		resultHeap := make(entity.ResultHeap, 0, k)
		heap.Init(&resultHeap)
		numWorkers := runtime.NumCPU()
		workChan := make(chan string, len(db.vectors))
		innerResultChan := make(chan entity.Result, len(db.vectors))
		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for id := range workChan {
					vec := db.vectors[id]
					sim := acceler.CosineSimilarity(normalizedQuery, vec)
					innerResultChan <- entity.Result{Id: id, Similarity: sim}
				}
			}()
		}
		go func() {
			for id := range db.vectors {
				workChan <- id
			}
			close(workChan)
		}()
		go func() {
			wg.Wait()
			close(innerResultChan)
		}()
		for res := range innerResultChan {
			if len(resultHeap) < k {
				heap.Push(&resultHeap, res)
			} else if res.Similarity > resultHeap[0].Similarity {
				heap.Pop(&resultHeap)
				heap.Push(&resultHeap, res)
			}
		}
		results = make([]entity.Result, len(resultHeap))
		for i := len(resultHeap) - 1; i >= 0; i-- {
			results[i] = heap.Pop(&resultHeap).(entity.Result)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Similarity > results[j].Similarity
		})
	}
	queryTime := time.Since(startTime)
	db.statsMu.Lock()
	alpha := 0.1
	db.stats.AvgQueryTime = time.Duration(float64(db.stats.AvgQueryTime)*(1-alpha) + float64(queryTime)*alpha)
	db.statsMu.Unlock()
	if k > len(results) {
		k = len(results)
	}

	return results[:k], nil
}

// HybridSearch 混合搜索策略
func (db *VectorDB) HybridSearch(query []float64, k int, options entity.SearchOptions, nprobe int) ([]entity.Result, error) {
	// 根据向量维度和数据规模自动选择最佳搜索策略
	if len(db.vectors) < 1000 || !db.indexed {
		// 小数据集使用暴力搜索
		return db.bruteForceSearch(query, k)
	} else if len(query) > 1000 {
		// 高维向量使用LSH (Locality-Sensitive Hashing)
		return db.lshSearch(query, k, options.NumHashTables)
	} else if db.config.UseMultiLevelIndex && db.multiIndex != nil && db.multiIndex.Indexed {
		return db.multiIndexSearch(query, k, nprobe)
	} else {
		// 默认使用IVF索引
		return db.ivfSearch(query, k, options.Nprobe)
	}
}

// bruteForceSearch 实现暴力搜索策略
// 适用于小数据集或索引未构建的情况
func (db *VectorDB) bruteForceSearch(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("vector is nil")
	}

	// 预处理查询向量 - 归一化可以提高相似度计算的准确性
	normalizedQuery := acceler.NormalizeVector(query)

	// 使用优先队列维护k个最近的向量
	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	// 使用工作池并行处理向量比较
	numWorkers := runtime.NumCPU()
	workChan := make(chan string, len(db.vectors))
	resultChan := make(chan entity.Result, len(db.vectors))

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for id := range workChan {
				vec := db.vectors[id]
				// 使用余弦相似度计算
				sim := acceler.CosineSimilarity(normalizedQuery, vec)
				resultChan <- entity.Result{Id: id, Similarity: sim}
			}
		}()
	}

	// 发送工作
	go func() {
		for id := range db.vectors {
			workChan <- id
		}
		close(workChan)
	}()

	// 等待所有工作完成并关闭结果通道
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 收集结果
	for result := range resultChan {
		if len(resultHeap) < k {
			heap.Push(&resultHeap, result)
		} else if result.Similarity > resultHeap[0].Similarity {
			heap.Pop(&resultHeap)
			heap.Push(&resultHeap, result)
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, min(k, len(resultHeap)))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// lshSearch 实现局部敏感哈希搜索
// 适用于高维向量搜索
func (db *VectorDB) lshSearch(query []float64, k int, numTables int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("vector is nil")
	}

	// 如果LSH索引未构建，则动态构建
	lshTables, err := db.buildLSHIndex(numTables)
	if err != nil {
		// 如果构建LSH索引失败，回退到暴力搜索
		logger.Error("构建LSH索引失败: %v，回退到暴力搜索\n", err)
		return db.bruteForceSearch(query, k)
	}

	// 预处理查询向量
	normalizedQuery := acceler.NormalizeVector(query)

	// 使用LSH索引进行搜索
	candidateSet := make(map[string]struct{})

	// 对每个哈希表进行查询
	for _, table := range lshTables {
		// 计算查询向量的哈希值
		hashValue := db.computeLSHHash(normalizedQuery, table.HashFunctions)

		// 获取对应哈希桶中的向量ID
		if ids, exists := table.Buckets[hashValue]; exists {
			for _, id := range ids {
				candidateSet[id] = struct{}{}
			}
		}
	}

	// 如果候选集为空，回退到暴力搜索
	if len(candidateSet) == 0 {
		return db.bruteForceSearch(query, k)
	}

	// 对候选集中的向量计算精确距离
	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	for id := range candidateSet {
		vec, exists := db.vectors[id]
		if !exists {
			continue
		}

		// 计算余弦相似度
		sim := acceler.CosineSimilarity(normalizedQuery, vec)

		if len(resultHeap) < k {
			heap.Push(&resultHeap, entity.Result{Id: id, Similarity: sim})
		} else if sim > resultHeap[0].Similarity {
			heap.Pop(&resultHeap)
			heap.Push(&resultHeap, entity.Result{Id: id, Similarity: sim})
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, min(k, len(resultHeap)))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// buildLSHIndex 构建LSH索引
func (db *VectorDB) buildLSHIndex(numTables int) ([]LSHTable, error) {
	if numTables <= 0 {
		numTables = 5 // 默认使用5个哈希表
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("数据库为空，无法构建LSH索引")
	}

	// 获取向量维度
	var dim int
	for _, vec := range db.vectors {
		dim = len(vec)
		break
	}

	// 创建LSH表
	tables := make([]LSHTable, numTables)

	// 为每个表生成随机哈希函数
	for i := range tables {
		// 每个表使用8个哈希函数
		numHashFunctions := 8
		hashFunctions := make([][]float64, numHashFunctions)

		// 生成随机哈希函数参数
		for j := range hashFunctions {
			hashFunctions[j] = make([]float64, dim)
			for k := range hashFunctions[j] {
				// 使用标准正态分布生成随机向量
				hashFunctions[j][k] = rand.NormFloat64()
			}
		}

		// 初始化哈希表
		tables[i] = LSHTable{
			HashFunctions: hashFunctions,
			Buckets:       make(map[uint64][]string),
		}

		// 将所有向量添加到哈希表中
		for id, vec := range db.vectors {
			hashValue := db.computeLSHHash(vec, hashFunctions)
			tables[i].Buckets[hashValue] = append(tables[i].Buckets[hashValue], id)
		}
	}

	return tables, nil
}

// computeLSHHash 计算向量的LSH哈希值
func (db *VectorDB) computeLSHHash(vec []float64, hashFunctions [][]float64) uint64 {
	// 使用随机超平面哈希
	hashBits := make([]uint64, len(hashFunctions))

	// 计算每个哈希函数的结果
	for i, hashFunc := range hashFunctions {
		// 计算向量与哈希函数的点积
		dotProduct := 0.0
		for j := range vec {
			if j < len(hashFunc) { // 确保索引有效
				dotProduct += vec[j] * hashFunc[j]
			}
		}

		// 如果点积大于0，哈希位为1，否则为0
		if dotProduct > 0 {
			hashBits[i] = 1
		}
	}

	// 将哈希位组合成一个64位整数
	var hashValue uint64
	for i, bit := range hashBits {
		if i < 64 { // 最多使用64位
			hashValue |= bit << i
		}
	}

	return hashValue
}

// 多级索引搜索
func (db *VectorDB) multiIndexSearch(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var results *entity.ResultHeap
	logger.Trace("Using Multi-Level Index for FindNearest.")
	// 1. 找到 nprobe 个最近的簇中心 (与之前逻辑类似)
	clusterDist := make([]struct {
		Index int
		Dist  float64
	}, len(db.multiIndex.Clusters))

	selectedStrategy := db.GetSelectStrategy()
	for i, cluster := range db.multiIndex.Clusters {
		dist, err := acceler.AdaptiveEuclideanDistanceSquared(query, cluster.Centroid, selectedStrategy)
		if err != nil {
			return nil, fmt.Errorf("error calculating distance to centroid %d: %w", i, err)
		}
		clusterDist[i] = struct {
			Index int
			Dist  float64
		}{i, dist}
	}

	sort.Slice(clusterDist, func(i, j int) bool {
		return clusterDist[i].Dist < clusterDist[j].Dist
	})

	numToProbe := nprobe
	if numToProbe > len(clusterDist) {
		numToProbe = len(clusterDist)
	}

	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	// 2. 在选中的簇的二级索引中搜索
	for i := 0; i < numToProbe; i++ {
		clusterIdx := clusterDist[i].Index
		selectedCluster := db.multiIndex.Clusters[clusterIdx]

		if clusterIdx >= len(db.multiIndex.SubIndices) || db.multiIndex.SubIndices[clusterIdx] == nil {
			logger.Warning("Sub-index for cluster %d not found or nil. Performing brute-force in this cluster.", clusterIdx)
			// 回退到暴力搜索该簇内的向量
			for _, id := range selectedCluster.VectorIDs {
				if vec, exists := db.vectors[id]; exists {
					dist, _ := acceler.AdaptiveEuclideanDistanceSquared(query, vec, selectedStrategy)
					heap.Push(&resultHeap, entity.Result{Id: id, Similarity: dist})
					if results.Len() > k {
						heap.Pop(results)
					}
				}
			}
			continue
		}

		// 假设二级索引是 KDTree，并且有 FindNearest 方法
		kdTree, ok := db.multiIndex.SubIndices[clusterIdx].(*tree.KDTree) // 类型断言
		if !ok || kdTree == nil {
			logger.Warning("Sub-index for cluster %d is not a KDTree or is nil. Performing brute-force.", clusterIdx)
			for _, id := range selectedCluster.VectorIDs {
				if vec, exists := db.vectors[id]; exists {
					dist, _ := acceler.AdaptiveEuclideanDistanceSquared(query, vec, selectedStrategy)
					heap.Push(&resultHeap, entity.Result{Id: id, Similarity: dist})
					if results.Len() > k {
						heap.Pop(results)
					}
				}
			}
			continue
		}

		kdResults := kdTree.FindNearest(query, k) // 在KD树中搜索K个最近的，或者一个合理的数量
		if kdResults == nil {
			logger.Error("Error searching in KDTree for cluster %d. Skipping this sub-index.", clusterIdx)
			continue
		}

		for _, item := range kdResults {
			heap.Push(results, &item)
			if results.Len() > k {
				heap.Pop(results)
			}
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, min(k, len(resultHeap)))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// ivfSearch 实现倒排文件索引搜索
// 适用于已构建索引的一般情况
func (db *VectorDB) ivfSearch(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("vector is nil")
	}

	// 检查索引状态
	if !db.indexed || len(db.clusters) == 0 || db.numClusters <= 0 {
		// 如果索引未构建，回退到暴力搜索
		logger.Warning("索引未构建，回退到暴力搜索")
		return db.bruteForceSearch(query, k)
	}

	// 预处理查询向量
	normalizedQuery := acceler.NormalizeVector(query)

	// 设置默认nprobe值
	if nprobe <= 0 {
		nprobe = 1 // 默认搜索最近的一个簇
	}
	if nprobe > db.numClusters {
		nprobe = db.numClusters // 不能超过总簇数
	}

	// 使用堆结构来维护最近的nprobe个簇
	centroidHeap := make(entity.CentroidHeap, 0, db.numClusters)

	// 获取最佳计算策略
	selectedStrategy := db.GetSelectStrategy()

	// 找到查询向量最近的nprobe个簇中心
	for i, cluster := range db.clusters {
		distSq, err := acceler.AdaptiveEuclideanDistanceSquared(normalizedQuery, cluster.Centroid, selectedStrategy)
		if err != nil {
			continue
		}
		centroidHeap = append(centroidHeap, entity.CentroidDist{Index: i, Distance: distSq})
	}

	// 堆化并提取最近的nprobe个簇
	heap.Init(&centroidHeap)
	var nearestClusters []int
	for i := 0; i < nprobe && len(centroidHeap) > 0; i++ {
		nearestClusters = append(nearestClusters, heap.Pop(&centroidHeap).(entity.CentroidDist).Index)
	}

	// 使用优先队列维护k个最近的向量
	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	// 在这nprobe个簇中搜索向量
	for _, clusterIndex := range nearestClusters {
		selectedCluster := db.clusters[clusterIndex]

		// 批量处理每个簇中的向量
		batchSize := 100
		for i := 0; i < len(selectedCluster.VectorIDs); i += batchSize {
			end := i + batchSize
			if end > len(selectedCluster.VectorIDs) {
				end = len(selectedCluster.VectorIDs)
			}

			// 并行处理每批向量
			var wg sync.WaitGroup
			resultChan := make(chan entity.Result, end-i)

			for j := i; j < end; j++ {
				wg.Add(1)
				go func(vecID string) {
					defer wg.Done()
					vec, exists := db.vectors[vecID]
					if !exists {
						return
					}
					// 计算余弦相似度
					sim := acceler.CosineSimilarity(normalizedQuery, vec)
					resultChan <- entity.Result{Id: vecID, Similarity: sim}
				}(selectedCluster.VectorIDs[j])
			}

			// 等待所有goroutine完成
			go func() {
				wg.Wait()
				close(resultChan)
			}()

			// 收集结果并维护最大堆
			for result := range resultChan {
				if len(resultHeap) < k {
					heap.Push(&resultHeap, result)
				} else if result.Similarity > resultHeap[0].Similarity {
					heap.Pop(&resultHeap)
					heap.Push(&resultHeap, result)
				}
			}
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, min(k, len(resultHeap)))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// FileSystemSearch 优化后的FileSystemSearch方法
func (db *VectorDB) FileSystemSearch(query string, vectorizedType int, k int, nprobe int) ([]entity.Result, error) {
	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	// 将查询转换为向量
	var vectorized DocumentVectorized
	switch vectorizedType {
	case TfidfVectorized:
		vectorized = TFIDFVectorized()
	case SimpleVectorized:
		vectorized = SimpleBagOfWordsVectorized()
	case WordEmbeddingVectorized:
		embeddings, err := LoadWordEmbeddings("path/to/pretrained_embeddings.txt")
		if err != nil {
			return nil, fmt.Errorf("加载词向量文件失败: %w", err)
		}
		vectorized = EnhancedWordEmbeddingVectorized(embeddings)
	default:
		return nil, fmt.Errorf("不支持的向量化类型: %d", vectorizedType)
	}

	queryVector, err := vectorized(query)
	if err != nil {
		return nil, fmt.Errorf("将查询转换为向量时出错: %w", err)
	}

	// 提取查询关键词，用于倒排索引过滤
	words := strings.Fields(query)
	if len(words) == 0 {
		return nil, fmt.Errorf("查询为空")
	}

	// 使用读锁保护并发访问
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 首先使用倒排索引快速筛选候选集
	candidateSet := make(map[string]int)
	for _, word := range words {
		if docIDs, exists := db.invertedIndex[word]; exists {
			for _, id := range docIDs {
				candidateSet[id]++
			}
		}
	}

	// 如果候选集为空，直接使用向量搜索
	if len(candidateSet) == 0 {
		if nprobe > 0 {
			return db.FindNearest(queryVector, k, nprobe)
		} else {
			return db.ParallelFindNearest(queryVector, k)
		}
	}

	// 根据匹配关键词数量排序候选集
	type candidate struct {
		id    string
		count int
	}
	candidates := make([]candidate, 0, len(candidateSet))
	for id, count := range candidateSet {
		candidates = append(candidates, candidate{id, count})
	}

	// 按匹配关键词数量降序排序
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].count > candidates[j].count
	})

	// 限制候选集大小，只保留匹配度最高的前N个
	maxCandidates := 1000
	if len(candidates) > maxCandidates {
		candidates = candidates[:maxCandidates]
	}

	// 对筛选后的候选集进行向量相似度排序
	// 使用工作池并行计算相似度
	numWorkers := runtime.NumCPU()
	workChan := make(chan candidate, len(candidates))
	resultChan := make(chan entity.Result, len(candidates))

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range workChan {
				vec, exists := db.vectors[c.id]
				if !exists {
					continue
				}
				sim := acceler.CosineSimilarity(queryVector, vec)
				resultChan <- entity.Result{Id: c.id, Similarity: sim, WordCount: c.count}
			}
		}()
	}

	// 发送工作
	go func() {
		for _, c := range candidates {
			workChan <- c
		}
		close(workChan)
	}()

	// 等待所有工作完成并关闭结果通道
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 收集结果
	results := make([]entity.Result, 0, len(candidates))
	for r := range resultChan {
		results = append(results, r)
	}

	// 使用混合排序：先按关键词匹配数量，再按向量相似度
	sort.Slice(results, func(i, j int) bool {
		// 如果关键词匹配数量相差超过阈值，优先考虑匹配数量
		if math.Abs(float64(results[i].WordCount-results[j].WordCount)) > 2 {
			return results[i].WordCount > results[j].WordCount
		}
		// 否则按相似度排序
		return results[i].Similarity > results[j].Similarity
	})

	// 提取前k个结果
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// 改进的查询缓存结构
type enhancedQueryCache struct {
	results    []string  // 结果ID列表
	timestamp  time.Time // 缓存创建时间
	vectorHash uint64    // 查询向量的哈希值
}

// ParallelFindNearest 优化的并行查询实现
func (db *VectorDB) ParallelFindNearest(query []float64, k int) ([]entity.Result, error) {
	// 创建固定大小的工作池
	numWorkers := runtime.NumCPU()

	// 使用更高效的任务分配策略
	vectorIDs := make([]string, 0, len(db.vectors))
	for id := range db.vectors {
		vectorIDs = append(vectorIDs, id)
	}

	// 计算每个工作协程处理的向量数量
	vectorsPerWorker := (len(vectorIDs) + numWorkers - 1) / numWorkers

	// 创建结果通道
	resultChan := make(chan []entity.Result, numWorkers)

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// 计算此工作协程处理的向量范围
			startIdx := workerID * vectorsPerWorker
			endIdx := startIdx + vectorsPerWorker
			if endIdx > len(vectorIDs) {
				endIdx = len(vectorIDs)
			}

			// 跳过空范围
			if startIdx >= len(vectorIDs) {
				resultChan <- nil
				return
			}

			// 处理分配的向量
			localResults := make([]entity.Result, 0, endIdx-startIdx)
			for idx := startIdx; idx < endIdx; idx++ {
				id := vectorIDs[idx]
				vec := db.vectors[id]
				sim := acceler.OptimizedCosineSimilarity(query, vec)
				localResults = append(localResults, entity.Result{Id: id, Similarity: sim})
			}

			// 本地排序，减少全局合并开销
			sort.Slice(localResults, func(i, j int) bool {
				return localResults[i].Similarity > localResults[j].Similarity
			})

			// 只保留前k个结果
			if len(localResults) > k {
				localResults = localResults[:k]
			}

			resultChan <- localResults
		}(i)
	}

	// 等待所有工作协程完成并关闭结果通道
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 合并结果
	allResults := make([]entity.Result, 0, k*numWorkers)
	for results := range resultChan {
		if results != nil {
			allResults = append(allResults, results...)
		}
	}

	// 全局排序并截取前k个结果
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Similarity > allResults[j].Similarity
	})

	if len(allResults) > k {
		allResults = allResults[:k]
	}

	return allResults, nil
}

func (db *VectorDB) GetDataSize() int {
	return len(db.vectors)
}

// GetPCAConfig 获取PCA配置
func (db *VectorDB) GetPCAConfig() *PCAConfig {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.pcaConfig
}

// CheckIndexHealth 添加索引健康检查方法
func (db *VectorDB) CheckIndexHealth() map[string]bool {
	db.mu.RLock()
	defer db.mu.RUnlock()

	health := make(map[string]bool)

	// 获取GPU加速器
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()

	// 检查EnhancedIVF索引健康状态
	health["enhanced_ivf"] = db.ivfIndex != nil && db.ivfIndex.Enable && db.ivfConfig != nil

	// 检查EnhancedLSH索引健康状态
	health["enhanced_lsh"] = db.LshIndex != nil && db.LshIndex.Enable && db.LshConfig != nil

	// 检查IVF-HNSW混合索引健康状态
	health["ivf_hnsw"] = db.ivfHnswIndex != nil && db.ivfHnswIndex.Enable && db.ivfHnswConfig != nil && db.useIVFHNSWIndex

	// 检查传统索引健康状态
	health["traditional_ivf"] = db.indexed && len(db.clusters) > 0
	health["hnsw"] = db.useHNSWIndex && db.indexed && db.hnsw != nil
	health["pq"] = db.usePQCompression && db.pqCodebook != nil
	health["gpu"] = db.HardwareCaps.HasGPU && gpuAccelerator != nil

	return health
}

// EnableMultiLevelCache 启用多级缓存
func (db *VectorDB) EnableMultiLevelCache(l1Size, l2Size, l3Size int, l3Path string) {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.MultiCache = NewMultiLevelCache(l1Size, l2Size, l3Size, l3Path)
	logger.Info("多级缓存已启用：L1=%d, L2=%d, L3=%s", l1Size, l2Size, l3Path)
}

// CachedSearch 带缓存的搜索
func (db *VectorDB) CachedSearch(query []float64, k int) ([]entity.Result, error) {
	// 生成查询键
	queryKey := db.generateQueryKey(query, k)

	// 尝试从缓存获取
	if db.MultiCache != nil {
		if cachedData, found := db.MultiCache.Get(queryKey); found {
			// 缓存命中，需要进行类型断言
			if cachedIDs, ok := cachedData.([]string); ok {
				// 类型断言成功，转换为 Result 格式
				results := make([]entity.Result, len(cachedIDs))
				for i, id := range cachedIDs {
					// 重新计算相似度（或从缓存中获取）
					similarity, err := db.CalculateCosineSimilarity(id, query)
					if err != nil {
						logger.Warning("计算向量 %s 的相似度失败: %v", id, err)
						continue
					}
					results[i] = entity.Result{
						Id:         id,
						Similarity: similarity,
					}
				}
				// 过滤掉可能的空结果（由于相似度计算失败）
				validResults := make([]entity.Result, 0, len(results))
				for _, result := range results {
					if result.Id != "" {
						validResults = append(validResults, result)
					}
				}
				return validResults, nil
			} else {
				logger.Warning("缓存数据类型断言失败，预期[]string，实际%T", cachedData)
				// 类型断言失败，继续执行搜索
			}
		}
	}

	// 缓存未命中，执行搜索
	results, err := db.TwoStageSearch(query, TwoStageSearchConfig{
		CoarseK:      k * 3,
		CoarseNprobe: db.numClusters / 5,
		FineK:        k,
		UseGPU:       false,
		Threshold:    0.1,
	})

	if err != nil {
		return nil, err
	}

	// 将结果存入缓存
	if db.MultiCache != nil {
		ids := make([]string, len(results))
		for i, result := range results {
			ids[i] = result.Id
		}
		db.MultiCache.Put(queryKey, ids)
	}

	return results, nil
}

// generateQueryKey 生成查询键
func (db *VectorDB) generateQueryKey(query []float64, k int) string {
	h := fnv.New64a()
	for _, val := range query {
		err := binary.Write(h, binary.LittleEndian, val)
		if err != nil {
			return ""
		}
	}
	err := binary.Write(h, binary.LittleEndian, int64(k))
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%x", h.Sum64())
}

// OptimizedBatchSearch GPU加速的批量搜索方法
func (db *VectorDB) OptimizedBatchSearch(queries [][]float64, k int, options entity.SearchOptions) ([][]entity.Result, error) {
	// 更新查询计数
	db.statsMu.Lock()
	db.stats.TotalQueries += int64(len(queries))
	db.statsMu.Unlock()

	db.mu.RLock()
	defer db.mu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		// 如果数据库为空，返回空结果
		emptyResults := make([][]entity.Result, len(queries))
		for i := range emptyResults {
			emptyResults[i] = []entity.Result{}
		}
		return emptyResults, nil
	}

	// 检查是否可以使用GPU加速
	if db.shouldUseGPUBatchSearch(len(queries), len(db.vectors)) {
		logger.Info("使用GPU加速批量搜索，查询数量: %d, 数据库大小: %d", len(queries), len(db.vectors))
		return db.gpuBatchSearch(queries, k, options)
	}

	// 回退到传统批量搜索方法
	logger.Trace("使用传统批量搜索方法")
	return db.fallbackBatchSearch(queries, k, options)
}

// shouldUseGPUBatchSearch 判断是否应该使用GPU批量搜索
func (db *VectorDB) shouldUseGPUBatchSearch(queryCount, dbSize int) bool {
	// 获取GPU加速器
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	// 检查GPU是否可用
	if !db.HardwareCaps.HasGPU || gpuAccelerator == nil {
		return false
	}

	// 检查GPU加速器是否已初始化
	if gpuAccel, ok := gpuAccelerator.(*acceler.FAISSAccelerator); ok {
		if err := gpuAccel.CheckGPUAvailability(); err != nil {
			logger.Warning("GPU不可用，回退到CPU搜索: %v", err)
			return false
		}
	} else {
		return false
	}

	// 获取向量维度
	vectorDim := db.vectorDim

	// 计算总计算量：查询数量 * 数据库大小 * 向量维度
	totalComputeLoad := int64(queryCount) * int64(dbSize) * int64(vectorDim)

	// 获取GPU配置（如果有硬件管理器）
	var batchSize int = 1000 // 默认批处理大小
	if db.hardwareManager != nil {
		config := db.hardwareManager.GetConfig()
		if config != nil && config.GPU.Enable {
			batchSize = config.GPU.BatchSize
		}
	}

	// 基于计算量的智能决策
	// 1. 小规模计算（查询少且数据库小）：使用CPU
	if queryCount < 5 && dbSize < 1000 {
		return false
	}

	// 2. 中等规模计算：根据向量维度决定
	if totalComputeLoad >= 10_000_000 { // 1千万次浮点运算
		logger.Info("使用GPU加速批量搜索，计算量较大: %d (查询:%d x 数据库:%d x 维度:%d)",
			totalComputeLoad, queryCount, dbSize, vectorDim)
		return true
	}

	// 3. 批量查询优化：当查询数量接近或超过批处理大小时
	if queryCount >= batchSize/10 {
		logger.Info("使用GPU加速批量搜索，查询批量较大: %d (批处理大小: %d)",
			queryCount, batchSize)
		return true
	}

	// 4. 大规模数据库：即使单个查询也值得使用GPU
	if dbSize >= 100000 {
		logger.Info("使用GPU加速批量搜索，数据库规模较大: %d", dbSize)
		return true
	}

	// 5. 高维向量处理：高维向量更适合GPU并行计算
	if vectorDim >= 512 && dbSize >= 10000 {
		logger.Info("使用GPU加速批量搜索，高维向量: %d", vectorDim)
		return true
	}

	return false
}

// gpuBatchSearch GPU加速的批量搜索实现
func (db *VectorDB) gpuBatchSearch(queries [][]float64, k int, options entity.SearchOptions) ([][]entity.Result, error) {
	// 记录开始时间，用于性能分析
	startTime := time.Now()

	// 记录搜索规模
	logger.Debug("开始GPU批量搜索，查询数量: %d, k值: %d, 数据库大小: %d", len(queries), k, len(db.vectors))

	// 获取GPU加速器
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	// 检查GPU加速器是否初始化
	if gpuAccelerator == nil {
		logger.Warning("GPU加速器未初始化，回退到CPU搜索")
		return db.fallbackBatchSearch(queries, k, options)
	}

	// 准备数据库向量数组
	database := make([][]float64, 0, len(db.vectors))
	idMapping := make([]string, 0, len(db.vectors))

	// 构建向量数组和ID映射
	for id, vector := range db.vectors {
		database = append(database, vector)
		idMapping = append(idMapping, id)
	}

	// 调用GPU加速器的BatchSearch方法
	gpuResults, err := gpuAccelerator.BatchSearch(queries, database, k)

	// 计算耗时
	elapsed := time.Since(startTime)

	// 错误处理和回退逻辑
	if err != nil {
		// 记录详细错误信息
		logger.Warning("GPU批量搜索失败(耗时: %v)，回退到CPU搜索: %v", elapsed, err)

		// 检查是否是内存不足错误或CUDA错误
		if strings.Contains(err.Error(), "memory") || strings.Contains(err.Error(), "CUDA") {
			logger.Error("GPU内存不足或CUDA错误，考虑调整批处理大小或使用CPU搜索")

			// 更新统计信息
			db.statsMu.Lock()
			db.stats.GPUErrors++
			db.statsMu.Unlock()
		}

		// 回退到CPU搜索
		cpuStartTime := time.Now()
		cpuResults, cpuErr := db.fallbackBatchSearch(queries, k, options)
		cpuElapsed := time.Since(cpuStartTime)

		if cpuErr != nil {
			logger.Error("CPU回退搜索也失败: %v", cpuErr)
			return nil, fmt.Errorf("GPU和CPU搜索均失败: %v, %v", err, cpuErr)
		}

		logger.Info("CPU回退搜索成功，耗时: %v (GPU尝试耗时: %v)", cpuElapsed, elapsed)
		return cpuResults, nil
	}

	// 转换GPU结果为entity.Result格式
	results := make([][]entity.Result, len(gpuResults))
	for i, queryResults := range gpuResults {
		results[i] = make([]entity.Result, min(k, len(queryResults)))
		for j, gpuResult := range queryResults {
			// 将GPU返回的数字ID转换为实际的向量ID
			gpuID := gpuResult.ID
			if idx, err := strconv.Atoi(gpuID); err == nil && idx < len(idMapping) {
				results[i][j] = entity.Result{
					Id:         idMapping[idx],
					Similarity: gpuResult.Similarity,
				}
			} else {
				// 如果ID转换失败，使用原始ID
				logger.Warning("GPU结果ID转换失败: %s", gpuID)
				results[i][j] = entity.Result{
					Id:         gpuID,
					Similarity: gpuResult.Similarity,
				}
			}
		}
	}

	// 计算成功，记录性能信息
	logger.Debug("GPU批量搜索完成，耗时: %v，结果数量: %d", elapsed, len(results))

	// 更新统计信息
	db.statsMu.Lock()
	db.stats.GPUSearches++
	db.stats.TotalGPUSearchTime += elapsed.Milliseconds()
	db.statsMu.Unlock()

	return results, nil
}

// fallbackBatchSearch 传统批量搜索方法（回退方案）
func (db *VectorDB) fallbackBatchSearch(queries [][]float64, k int, options entity.SearchOptions) ([][]entity.Result, error) {
	// 使用现有的BatchFindNearestWithScores方法
	numWorkers := runtime.NumCPU()
	return db.BatchFindNearestWithScores(queries, k, numWorkers)
}

// IncrementalIndex 执行增量索引更新，根据当前索引类型选择合适的增量更新方法
func (db *VectorDB) IncrementalIndex() error {
	// 记录开始时间，用于性能统计
	start := time.Now()

	// 检查是否有索引
	db.mu.Lock()
	if !db.indexed {
		db.mu.Unlock()
		logger.Warning("索引尚未构建，无法执行增量索引更新")
		return fmt.Errorf("索引尚未构建，请先调用 BuildIndex() 或 RebuildIndex()")
	}

	// 获取所有向量ID
	var vectorIDs []string
	for id := range db.vectors {
		vectorIDs = append(vectorIDs, id)
	}
	db.mu.Unlock()

	logger.Info("开始执行增量索引更新，共 %d 个向量...", len(vectorIDs))

	// 根据索引类型选择不同的增量更新方法
	if db.useHNSWIndex {
		// 使用HNSW增量更新
		for _, id := range vectorIDs {
			db.mu.Lock()
			vector, exists := db.vectors[id]
			db.mu.Unlock()

			if !exists {
				continue
			}

			err := db.UpdateHNSWIndexIncrementally(id, vector)
			if err != nil {
				logger.Warning("向量 %s 的HNSW增量更新失败: %v", id, err)
			}
		}
	} else {
		// 使用传统IVF增量更新
		for _, id := range vectorIDs {
			db.mu.Lock()
			vector, exists := db.vectors[id]
			db.mu.Unlock()

			if !exists {
				continue
			}

			err := db.UpdateIndexIncrementally(id, vector)
			if err != nil {
				logger.Warning("向量 %s 的IVF增量更新失败: %v", id, err)
			}
		}
	}

	// 更新性能统计信息
	db.statsMu.Lock()
	db.stats.LastReindexTime = time.Now()

	// 更新内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	db.stats.MemoryUsage = m.Alloc
	db.statsMu.Unlock()

	logger.Info("增量索引更新完成，耗时: %v", time.Since(start))
	return nil
}

// OptimizeIndex 根据当前索引类型执行索引优化
func (db *VectorDB) OptimizeIndex() error {
	// 获取索引健康状态
	indexHealth := db.CheckIndexHealth()

	// 记录开始时间，用于性能统计
	start := time.Now()

	// 根据索引类型执行不同的优化策略
	if indexHealth["hnsw"] {
		// 优化HNSW索引参数
		logger.Info("优化HNSW索引参数...")
		db.OptimizeHNSWParameters()
	}

	// 如果启用了增强型LSH索引，执行LSH参数调优
	if indexHealth["enhanced_lsh"] {
		logger.Info("优化LSH索引参数...")
		db.tuneAdaptiveLSH()
	}

	// 如果启用了增强型IVF索引，执行IVF参数调优
	if indexHealth["enhanced_ivf"] {
		logger.Info("优化IVF索引参数...")
		// 根据数据规模调整IVF参数
		db.mu.Lock()
		dataSize := len(db.vectors)
		db.mu.Unlock()

		// 调整nlist参数
		if db.ivfConfig != nil {
			if dataSize < 10000 {
				db.ivfConfig.NumClusters = 100
			} else if dataSize < 100000 {
				db.ivfConfig.NumClusters = 256
			} else if dataSize < 1000000 {
				db.ivfConfig.NumClusters = 1024
			} else {
				db.ivfConfig.NumClusters = 4096
			}

			logger.Info("IVF参数已优化: NumClusters=%d", db.ivfConfig.NumClusters)
		}
	}

	// 如果启用了IVF-HNSW混合索引，执行参数调优
	if indexHealth["ivf_hnsw"] {
		logger.Info("优化IVF-HNSW混合索引参数...")
		db.optimizeIVFHNSWParameters()
	}

	// 如果启用了PQ压缩，优化PQ参数
	if indexHealth["pq"] {
		logger.Info("优化PQ压缩参数...")
		// 根据向量维度调整子向量数量
		db.mu.Lock()
		vectorDim := db.vectorDim
		db.mu.Unlock()

		// 调整子向量数量，确保能被维度整除
		if vectorDim >= 200 {
			db.numSubVectors = 16
		} else if vectorDim >= 100 {
			db.numSubVectors = 8
		} else if vectorDim >= 50 {
			db.numSubVectors = 4
		} else {
			db.numSubVectors = 2
		}

		// 确保子向量数量能被维度整除
		for vectorDim%db.numSubVectors != 0 {
			db.numSubVectors--
			if db.numSubVectors < 1 {
				db.numSubVectors = 1
				break
			}
		}

		logger.Info("PQ参数已优化: 子向量数量=%d", db.numSubVectors)
	}

	// 调整全局配置参数
	db.AdjustConfig()

	// 更新性能统计信息
	db.statsMu.Lock()
	// 更新内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	db.stats.MemoryUsage = m.Alloc
	db.statsMu.Unlock()

	logger.Info("索引优化完成，耗时: %v", time.Since(start))
	return nil
}

// ==================== IVF-HNSW 混合索引实现 ====================

// BuildIVFHNSWIndex 构建IVF-HNSW混合索引
func (db *VectorDB) BuildIVFHNSWIndex(config *IVFHNSWConfig) error {
	if config == nil {
		return fmt.Errorf("IVF-HNSW配置不能为空")
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if len(db.vectors) == 0 {
		return fmt.Errorf("数据库为空，无法构建索引")
	}

	logger.Info("开始构建IVF-HNSW混合索引，数据量: %d, 聚类数: %d", len(db.vectors), config.NumClusters)
	startTime := time.Now()

	// 检查配置参数的合理性
	if config.NumClusters <= 0 {
		return fmt.Errorf("聚类数必须大于0")
	}
	if config.NumClusters > len(db.vectors) {
		return fmt.Errorf("聚类数不能大于向量数量")
	}

	// 1. 初始化索引结构
	index := &IVFHNSWIndex{
		Clusters:         make([]IVFHNSWCluster, config.NumClusters),
		ClusterCentroids: make([][]float64, config.NumClusters),
		ClusterSizes:     make([]int, config.NumClusters),
		NumClusters:      config.NumClusters,
		ClusterGraphs:    make(map[int]*graph.HNSWGraph),
		TotalVectors:     len(db.vectors),
		IndexVersion:     1,
		LastUpdateTime:   time.Now(),
		Enable:           true,
	}

	// 2. 准备训练数据
	trainingData := db.sampleTrainingDataForIVFHNSW(config.TrainingRatio)
	if len(trainingData) < config.NumClusters {
		return fmt.Errorf("训练数据不足，需要至少 %d 个样本", config.NumClusters)
	}

	// 3. 使用K-means聚类算法生成聚类中心
	logger.Info("执行K-means聚类...")
	if len(trainingData) == 0 {
		return fmt.Errorf("训练数据为空")
	}
	centroids, err := db.performKMeansForIVFHNSW(trainingData, config.NumClusters)
	if err != nil {
		return fmt.Errorf("K-means聚类失败: %w", err)
	}
	if len(centroids) != config.NumClusters {
		return fmt.Errorf("聚类中心数量不匹配，期望: %d, 实际: %d", config.NumClusters, len(centroids))
	}

	// 4. 初始化聚类
	for i, centroid := range centroids {
		index.ClusterCentroids[i] = centroid
		index.Clusters[i] = IVFHNSWCluster{
			Centroid:     centroid,
			VectorIDs:    make([]string, 0),
			UseHNSW:      false,
			PQCodes:      make(map[string][]byte),
			LastAccessed: time.Now(),
			AccessCount:  0,
		}
	}

	// 5. 分配向量到聚类
	logger.Info("分配向量到聚类...")
	for id, vector := range db.vectors {
		clusterID := db.findNearestClusterForIVFHNSW(vector, centroids)
		index.Clusters[clusterID].VectorIDs = append(index.Clusters[clusterID].VectorIDs, id)
		index.ClusterSizes[clusterID]++
	}

	// 6. 为大聚类构建HNSW图
	logger.Info("为大聚类构建HNSW图...")
	for i := range index.Clusters {
		clusterSize := len(index.Clusters[i].VectorIDs)
		if clusterSize >= config.MinClusterSize {
			// 构建聚类内的HNSW图
			hnswGraph := graph.NewHNSWGraph(config.MaxConnections, config.EfConstruction, config.EfSearch)

			// 添加聚类内的向量到HNSW图
			successCount := 0
			for _, vectorID := range index.Clusters[i].VectorIDs {
				if vector, exists := db.vectors[vectorID]; exists {
					if err := hnswGraph.AddNode(vectorID, vector); err != nil {
						logger.Warning("添加向量到HNSW图失败: %v", err)
						continue
					}
					successCount++
				}
			}

			// 只有成功添加了足够的向量才使用HNSW
			if successCount >= config.MinClusterSize/2 {
				index.Clusters[i].HNSWGraph = hnswGraph
				index.Clusters[i].UseHNSW = true
				index.ClusterGraphs[i] = hnswGraph
				logger.Info("聚类 %d 构建HNSW图完成，向量数: %d/%d", i, successCount, clusterSize)
			} else {
				logger.Warning("聚类 %d HNSW图构建失败，成功向量数不足: %d/%d", i, successCount, clusterSize)
			}
		}
	}

	// 7. 构建全局HNSW图（聚类中心）
	logger.Info("构建全局HNSW图...")
	globalGraph := graph.NewHNSWGraph(config.MaxConnections, config.EfConstruction, config.EfSearch)
	globalSuccessCount := 0
	for i, centroid := range centroids {
		clusterID := fmt.Sprintf("cluster_%d", i)
		if err := globalGraph.AddNode(clusterID, centroid); err != nil {
			logger.Warning("添加聚类中心到全局HNSW图失败: %v", err)
		} else {
			globalSuccessCount++
		}
	}
	if globalSuccessCount > 0 {
		index.GlobalGraph = globalGraph
		logger.Info("全局HNSW图构建完成，成功添加聚类中心: %d/%d", globalSuccessCount, len(centroids))
	} else {
		logger.Warning("全局HNSW图构建失败，无法添加任何聚类中心")
	}

	// 8. 计算聚类指标
	db.calculateIVFHNSWClusterMetrics(index)

	// 9. 保存索引
	db.ivfHnswIndex = index
	db.ivfHnswConfig = config
	db.useIVFHNSWIndex = true

	buildTime := time.Since(startTime)
	logger.Info("IVF-HNSW混合索引构建完成，耗时: %v", buildTime)

	// 更新性能统计
	db.statsMu.Lock()
	db.stats.IndexBuildTime = buildTime
	db.stats.LastReindexTime = time.Now()
	db.statsMu.Unlock()

	return nil
}

// ivfHnswSearchWithScores IVF-HNSW混合搜索
func (db *VectorDB) ivfHnswSearchWithScores(query []float64, k int, ctx SearchContext) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.useIVFHNSWIndex || db.ivfHnswIndex == nil {
		return nil, fmt.Errorf("IVF-HNSW混合索引未启用或未构建")
	}

	if len(query) == 0 {
		return []entity.Result{}, nil
	}

	startTime := time.Now()
	index := db.ivfHnswIndex
	config := db.ivfHnswConfig

	// 1. 使用全局HNSW图找到最相关的聚类
	nprobe := config.Nprobe
	if nprobe <= 0 {
		nprobe = int(math.Sqrt(float64(config.NumClusters)))
	}
	if nprobe > config.NumClusters {
		nprobe = config.NumClusters
	}

	logger.Trace("IVF-HNSW搜索：探测聚类数=%d", nprobe)

	// 2. 找到最近的聚类
	candidateClusters := db.findCandidateClustersIVFHNSW(query, nprobe)

	// 3. 在候选聚类中搜索
	allResults := make([]entity.Result, 0)
	for _, clusterID := range candidateClusters {
		if clusterID >= len(index.Clusters) {
			continue
		}

		cluster := &index.Clusters[clusterID]
		cluster.AccessCount++
		cluster.LastAccessed = time.Now()

		var clusterResults []entity.Result
		var err error

		// 根据聚类大小选择搜索策略
		if cluster.UseHNSW && cluster.HNSWGraph != nil {
			// 使用HNSW搜索
			clusterResults, err = db.searchInClusterHNSW(query, k, cluster)
		} else {
			// 使用暴力搜索
			clusterResults, err = db.searchInClusterBruteForce(query, k, cluster)
		}

		if err != nil {
			logger.Warning("聚类 %d 搜索失败: %v", clusterID, err)
			continue
		}

		allResults = append(allResults, clusterResults...)
	}

	// 4. 合并和排序结果
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Similarity > allResults[j].Similarity
	})

	if len(allResults) > k {
		allResults = allResults[:k]
	}

	// 5. 更新性能统计
	latency := time.Since(startTime)
	db.updateIVFHNSWPerformanceStats(latency, len(allResults), true)

	logger.Trace("IVF-HNSW搜索完成，返回结果数: %d，耗时: %v", len(allResults), latency)
	return allResults, nil
}

// sampleTrainingDataForIVFHNSW 为IVF-HNSW采样训练数据
func (db *VectorDB) sampleTrainingDataForIVFHNSW(ratio float64) [][]float64 {
	if ratio <= 0 || ratio > 1 {
		ratio = 0.1 // 默认使用10%的数据
	}

	totalVectors := len(db.vectors)
	sampleSize := int(float64(totalVectors) * ratio)
	if sampleSize < 100 {
		sampleSize = totalVectors // 如果样本太少，使用全部数据
	}

	trainingData := make([][]float64, 0, sampleSize)
	count := 0
	step := totalVectors / sampleSize
	if step < 1 {
		step = 1
	}

	for _, vector := range db.vectors {
		if count%step == 0 {
			trainingData = append(trainingData, vector)
			if len(trainingData) >= sampleSize {
				break
			}
		}
		count++
	}

	return trainingData
}

// performKMeansForIVFHNSW 执行K-means聚类
func (db *VectorDB) performKMeansForIVFHNSW(data [][]float64, k int) ([][]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("训练数据为空")
	}
	if k <= 0 {
		return nil, fmt.Errorf("聚类数必须大于0")
	}
	if len(data) < k {
		return nil, fmt.Errorf("训练数据数量 %d 少于聚类数 %d", len(data), k)
	}

	dim := len(data[0])
	centroids := make([][]float64, k)

	// 随机初始化聚类中心
	rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, dim)
		copy(centroids[i], data[rand.Intn(len(data))])
	}

	maxIterations := 100
	tolerance := 1e-6

	for iter := 0; iter < maxIterations; iter++ {
		// 分配数据点到最近的聚类中心
		assignments := make([]int, len(data))
		for i, point := range data {
			minDist := math.Inf(1)
			bestCluster := 0
			for j, centroid := range centroids {
				dist, _ := algorithm.EuclideanDistanceSquared(point, centroid)
				if dist < minDist {
					minDist = dist
					bestCluster = j
				}
			}
			assignments[i] = bestCluster
		}

		// 更新聚类中心
		newCentroids := make([][]float64, k)
		clusterCounts := make([]int, k)
		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float64, dim)
		}

		for i, point := range data {
			clusterID := assignments[i]
			for j := 0; j < dim; j++ {
				newCentroids[clusterID][j] += point[j]
			}
			clusterCounts[clusterID]++
		}

		// 计算平均值
		converged := true
		for i := 0; i < k; i++ {
			if clusterCounts[i] > 0 {
				for j := 0; j < dim; j++ {
					newCentroids[i][j] /= float64(clusterCounts[i])
				}

				// 检查收敛性
				dist, _ := algorithm.EuclideanDistanceSquared(centroids[i], newCentroids[i])
				if dist > tolerance {
					converged = false
				}
			} else {
				// 如果聚类为空，重新随机初始化
				copy(newCentroids[i], data[rand.Intn(len(data))])
				converged = false
			}
		}

		centroids = newCentroids
		if converged {
			logger.Info("K-means收敛，迭代次数: %d", iter+1)
			break
		}
	}

	return centroids, nil
}

// findNearestClusterForIVFHNSW 找到最近的聚类
func (db *VectorDB) findNearestClusterForIVFHNSW(vector []float64, centroids [][]float64) int {
	minDist := math.Inf(1)
	bestCluster := 0

	for i, centroid := range centroids {
		dist, _ := algorithm.EuclideanDistanceSquared(vector, centroid)
		if dist < minDist {
			minDist = dist
			bestCluster = i
		}
	}

	return bestCluster
}

// findCandidateClustersIVFHNSW 找到候选聚类
func (db *VectorDB) findCandidateClustersIVFHNSW(query []float64, nprobe int) []int {
	index := db.ivfHnswIndex
	candidates := make([]struct {
		clusterID int
		distance  float64
	}, 0, len(index.ClusterCentroids))

	// 计算查询向量到所有聚类中心的距离
	for i, centroid := range index.ClusterCentroids {
		dist, _ := algorithm.EuclideanDistanceSquared(query, centroid)
		candidates = append(candidates, struct {
			clusterID int
			distance  float64
		}{i, dist})
	}

	// 按距离排序
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	// 返回最近的nprobe个聚类
	result := make([]int, 0, nprobe)
	for i := 0; i < nprobe && i < len(candidates); i++ {
		result = append(result, candidates[i].clusterID)
	}

	return result
}

// searchInClusterHNSW 在聚类中使用HNSW搜索
func (db *VectorDB) searchInClusterHNSW(query []float64, k int, cluster *IVFHNSWCluster) ([]entity.Result, error) {
	if cluster.HNSWGraph == nil {
		return nil, fmt.Errorf("聚类HNSW图未初始化")
	}

	// 使用HNSW图搜索
	hnswResults, err := cluster.HNSWGraph.Search(query, k)
	if err != nil {
		return nil, fmt.Errorf("HNSW搜索失败: %w", err)
	}

	// 转换结果格式
	results := make([]entity.Result, 0, len(hnswResults))
	for _, hnswResult := range hnswResults {
		// HNSW返回的是距离，转换为相似度
		similarity := 1.0 / (1.0 + hnswResult.Similarity)
		results = append(results, entity.Result{
			Id:         hnswResult.Id,
			Similarity: similarity,
		})
	}

	return results, nil
}

// searchInClusterBruteForce 在聚类中使用暴力搜索
func (db *VectorDB) searchInClusterBruteForce(query []float64, k int, cluster *IVFHNSWCluster) ([]entity.Result, error) {
	results := make([]entity.Result, 0, len(cluster.VectorIDs))

	// 计算查询向量与聚类中所有向量的相似度
	for _, vectorID := range cluster.VectorIDs {
		if vector, exists := db.vectors[vectorID]; exists {
			similarity := acceler.CosineSimilarity(query, vector)
			results = append(results, entity.Result{
				Id:         vectorID,
				Similarity: similarity,
			})
		}
	}

	// 按相似度降序排序
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// 返回top-k结果
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// calculateIVFHNSWClusterMetrics 计算聚类指标
func (db *VectorDB) calculateIVFHNSWClusterMetrics(index *IVFHNSWIndex) {
	for i := range index.Clusters {
		cluster := &index.Clusters[i]
		if len(cluster.VectorIDs) == 0 {
			continue
		}

		// 计算方差
		variance := 0.0
		for _, vectorID := range cluster.VectorIDs {
			if vector, exists := db.vectors[vectorID]; exists {
				dist, _ := algorithm.EuclideanDistanceSquared(vector, cluster.Centroid)
				variance += dist
			}
		}
		variance /= float64(len(cluster.VectorIDs))

		// 计算半径（最大距离）
		radius := 0.0
		for _, vectorID := range cluster.VectorIDs {
			if vector, exists := db.vectors[vectorID]; exists {
				dist, _ := algorithm.EuclideanDistanceSquared(vector, cluster.Centroid)
				if dist > radius {
					radius = dist
				}
			}
		}

		// 计算密度
		density := float64(len(cluster.VectorIDs)) / (1.0 + radius)

		cluster.Metrics = IVFHNSWClusterMetrics{
			Variance:       variance,
			Density:        density,
			Radius:         radius,
			QueryFrequency: 0.0,
			LastRebalance:  time.Now(),
		}
	}
}

// updateIVFHNSWPerformanceStats 更新IVF-HNSW性能统计
func (db *VectorDB) updateIVFHNSWPerformanceStats(latency time.Duration, resultCount int, isHybrid bool) {
	if db.ivfHnswIndex == nil {
		return
	}

	db.ivfHnswIndex.mu.Lock()
	defer db.ivfHnswIndex.mu.Unlock()

	stats := &db.ivfHnswIndex.PerformanceStats
	stats.TotalQueries++

	if isHybrid {
		stats.HybridQueries++
		// 更新混合查询平均延迟
		if stats.HybridQueries == 1 {
			stats.AvgHybridLatency = latency
		} else {
			stats.AvgHybridLatency = time.Duration(
				(int64(stats.AvgHybridLatency)*int64(stats.HybridQueries-1) + int64(latency)) / int64(stats.HybridQueries),
			)
		}
	}

	// 计算QPS
	if stats.TotalQueries > 0 {
		elapsedTime := time.Since(stats.LastUpdated)
		if elapsedTime > 0 {
			stats.ThroughputQPS = float64(stats.TotalQueries) / elapsedTime.Seconds()
		}
	}

	stats.LastUpdated = time.Now()
}

// optimizeIVFHNSWParameters 优化IVF-HNSW参数
func (db *VectorDB) optimizeIVFHNSWParameters() {
	if db.ivfHnswConfig == nil || db.ivfHnswIndex == nil {
		return
	}

	db.mu.Lock()
	dataSize := len(db.vectors)
	vectorDim := db.vectorDim
	db.mu.Unlock()

	config := db.ivfHnswConfig

	// 根据数据规模调整聚类数
	if dataSize < 10000 {
		config.NumClusters = 64
	} else if dataSize < 100000 {
		config.NumClusters = 256
	} else if dataSize < 1000000 {
		config.NumClusters = 1024
	} else {
		config.NumClusters = 4096
	}

	// 根据维度调整HNSW参数
	if vectorDim <= 128 {
		config.MaxConnections = 16
		config.EfConstruction = 200
		config.EfSearch = 50
	} else if vectorDim <= 512 {
		config.MaxConnections = 32
		config.EfConstruction = 400
		config.EfSearch = 100
	} else {
		config.MaxConnections = 64
		config.EfConstruction = 800
		config.EfSearch = 200
	}

	// 调整探测聚类数
	config.Nprobe = int(math.Sqrt(float64(config.NumClusters)))
	if config.Nprobe < 1 {
		config.Nprobe = 1
	}
	if config.Nprobe > config.NumClusters/4 {
		config.Nprobe = config.NumClusters / 4
	}

	// 调整聚类大小阈值
	config.MinClusterSize = 50
	config.MaxClusterSize = dataSize / config.NumClusters * 3

	logger.Info("IVF-HNSW参数已优化: NumClusters=%d, Nprobe=%d, MaxConnections=%d",
		config.NumClusters, config.Nprobe, config.MaxConnections)
}

// GetIVFHNSWStats 获取IVF-HNSW统计信息
func (db *VectorDB) GetIVFHNSWStats() *IVFHNSWPerformanceStats {
	if db.ivfHnswIndex == nil {
		return nil
	}

	db.ivfHnswIndex.mu.RLock()
	defer db.ivfHnswIndex.mu.RUnlock()

	stats := db.ivfHnswIndex.PerformanceStats
	return &stats
}

// InitializeIVFHNSWIndex 初始化IVF-HNSW混合索引
func (db *VectorDB) InitializeIVFHNSWIndex() error {
	// 添加panic恢复机制
	defer func() {
		if r := recover(); r != nil {
			logger.Error("IVF-HNSW索引初始化发生panic: %v", r)
		}
	}()

	// 检查数据库是否为空
	if len(db.vectors) == 0 {
		return fmt.Errorf("数据库为空，无法构建索引")
	}

	// 创建默认配置
	config := &IVFHNSWConfig{
		NumClusters:        256,
		TrainingRatio:      0.1,
		Nprobe:             16,
		RebalanceThreshold: 1000,
		MaxConnections:     32,
		EfConstruction:     400.0,
		EfSearch:           100.0,
		MaxLevel:           5,
		EnableHierarchical: true,
		ClusterHNSWRatio:   0.1,
		MinClusterSize:     50,
		MaxClusterSize:     10000,
		UsePQCompression:   false,
		PQSubVectors:       8,
		PQCentroids:        256,
	}

	// 根据数据规模调整配置
	db.mu.RLock()
	dataSize := len(db.vectors)
	vectorDim := db.vectorDim
	db.mu.RUnlock()

	// 检查数据量是否足够
	if dataSize < 10 {
		return fmt.Errorf("数据量不足，需要至少10个向量才能构建IVF-HNSW索引")
	}

	if dataSize > 0 {
		if dataSize < 1000 {
			config.NumClusters = min(dataSize/10, 16) // 确保聚类数不超过数据量的1/10
			if config.NumClusters < 4 {
				config.NumClusters = 4
			}
			config.Nprobe = 4
		} else if dataSize < 10000 {
			config.NumClusters = 64
			config.Nprobe = 8
		} else if dataSize < 100000 {
			config.NumClusters = 256
			config.Nprobe = 16
		} else if dataSize < 1000000 {
			config.NumClusters = 1024
			config.Nprobe = 32
		} else {
			config.NumClusters = 4096
			config.Nprobe = 64
		}
	}

	if vectorDim > 0 {
		if vectorDim <= 128 {
			config.MaxConnections = 16
			config.EfConstruction = 200.0
		} else if vectorDim <= 512 {
			config.MaxConnections = 32
			config.EfConstruction = 400.0
		} else {
			config.MaxConnections = 64
			config.EfConstruction = 800.0
		}
	}

	// 构建索引
	return db.BuildIVFHNSWIndex(config)
}
