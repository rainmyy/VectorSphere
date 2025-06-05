package db

import (
	"VectorSphere/library/algorithm"
	"VectorSphere/library/entity"
	"VectorSphere/library/graph"
	"VectorSphere/library/log"
	"VectorSphere/library/tree"
	"container/heap"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
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
)

// SearchContext 搜索上下文
type SearchContext struct {
	QueryVector  []float64
	K            int
	Nprobe       int
	Timeout      time.Duration
	QualityLevel float64 // 0.0-1.0，质量要求等级
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

// 添加查询缓存结构
type queryCache struct {
	results   []string
	timestamp int64
}

// PerformanceStats 性能统计结构
type PerformanceStats struct {
	TotalQueries    int64
	CacheHits       int64
	AvgQueryTime    time.Duration
	IndexBuildTime  time.Duration
	LastReindexTime time.Time
	MemoryUsage     uint64
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
	// 添加查询缓存
	queryCache map[string]queryCache
	cacheMu    sync.RWMutex
	cacheTTL   int64 // 缓存有效期（秒）

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
	numSubvectors            int              // PQ 的子向量数量
	numCentroidsPerSubvector int              // 每个子向量空间的质心数量
	usePQCompression         bool             // 标志是否启用 PQ 压缩

	stopCh           chan struct{}
	useNormalization bool
	hnsw             *graph.HNSWGraph // HNSW 图结构索引
	useHNSWIndex     bool             // 是否使用 HNSW 索引
	maxConnections   int              // HNSW 最大连接数
	efConstruction   float64          // HNSW 构建时的扩展因子
	efSearch         float64          // HNSW 搜索时的扩展因子
	metadata         map[string]map[string]interface{}

	multiCache     *MultiLevelCache         // 多级缓存
	gpuAccelerator algorithm.GPUAccelerator // GPU 加速器

	// 新增硬件自适应相关字段
	strategyComputeSelector *algorithm.ComputeStrategySelector
	currentStrategy         algorithm.ComputeStrategy
	hardwareCaps            algorithm.HardwareCapabilities
	strategySelector        *StrategySelector
}

const (
	DefaultVectorized = iota
	SimpleVectorized
	TfidfVectorized
	WordEmbeddingVectorized
)

// GetStats 获取性能统计信息
func (db *VectorDB) GetStats() PerformanceStats {
	db.statsMu.RLock()
	defer db.statsMu.RUnlock()
	return db.stats
}

// SelectOptimalIndexStrategy 智能选择最优索引策略
func (db *VectorDB) SelectOptimalIndexStrategy(ctx SearchContext) IndexStrategy {
	db.mu.RLock()
	vectorCount := len(db.vectors)
	vectorDim := len(ctx.QueryVector)
	db.mu.RUnlock()

	// 1. 数据规模判断
	if vectorCount < 1000 {
		return StrategyBruteForce
	}

	// 2. 维度判断
	if vectorDim > 2048 {
		// 高维数据优先考虑PQ压缩
		if db.usePQCompression && db.pqCodebook != nil {
			return StrategyPQ
		}
		// 其次考虑HNSW
		if db.useHNSWIndex && db.indexed && db.hnsw != nil {
			return StrategyHNSW
		}
	}

	// 3. 质量要求判断
	if ctx.QualityLevel > 0.9 {
		// 高质量要求，优先精确搜索
		if vectorCount < 100000 {
			return StrategyBruteForce
		}
		if db.useHNSWIndex && db.indexed && db.hnsw != nil {
			return StrategyHNSW
		}
	}

	// 4. 性能要求判断
	if ctx.Timeout > 0 && ctx.Timeout < 10*time.Millisecond {
		// 低延迟要求，优先快速策略
		if db.usePQCompression && db.pqCodebook != nil {
			return StrategyPQ
		}
	}

	// 5. 硬件能力判断
	if db.hardwareCaps.HasGPU && vectorCount > 50000 {
		return StrategyHybrid // GPU加速的混合策略
	}

	// 6. 默认策略选择
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		return StrategyHNSW
	}

	if db.indexed && len(db.clusters) > 0 {
		return StrategyIVF
	}

	return StrategyBruteForce
}

// OptimizedSearch 优化的搜索方法
func (db *VectorDB) OptimizedSearch(query []float64, k int, options SearchOptions) ([]entity.Result, error) {
	ctx := SearchContext{
		QueryVector:  query,
		K:            k,
		Nprobe:       options.Nprobe,
		Timeout:      options.SearchTimeout,
		QualityLevel: 0.8, // 默认质量等级
	}

	strategy := db.SelectOptimalIndexStrategy(ctx)

	log.Trace("选择搜索策略: %v, 数据量: %d, 维度: %d", strategy, len(db.vectors), len(query))

	startTime := time.Now()
	var results []entity.Result
	var err error

	switch strategy {
	case StrategyBruteForce:
		results, err = db.bruteForceSearch(query, k)
	case StrategyIVF:
		results, err = db.ivfSearchWithScores(query, k, ctx.Nprobe, db.GetSelectStrategy(query))
	case StrategyHNSW:
		results, err = db.hnswSearchWithScores(query, k)
	case StrategyPQ:
		results, err = db.pqSearchWithScores(query, k)
	case StrategyHybrid:
		results, err = db.hybridSearchWithScores(query, k, ctx)
	default:
		results, err = db.ivfSearchWithScores(query, k, ctx.Nprobe, db.GetSelectStrategy(query))
	}

	// 记录性能指标
	latency := time.Since(startTime)
	db.updatePerformanceMetrics(strategy, latency, len(results))

	return results, err
}

// pqSearchWithScores PQ压缩搜索
func (db *VectorDB) pqSearchWithScores(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.usePQCompression || db.pqCodebook == nil {
		return nil, fmt.Errorf("PQ压缩未启用")
	}

	results := make([]entity.Result, 0, len(db.compressedVectors))

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
					dist, err := db.CalculateApproximateDistancePQ(query, compVec)
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

// hnswSearchWithScores HNSW搜索并返回带分数的结果
func (db *VectorDB) hnswSearchWithScores(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if !db.useHNSWIndex || !db.indexed || db.hnsw == nil {
		return nil, fmt.Errorf("HNSW索引未启用或未构建")
	}

	if len(db.vectors) == 0 {
		return []entity.Result{}, nil
	}

	// 选择最优计算策略
	strategy := db.GetSelectStrategy(query)

	// 根据策略设置距离函数
	switch strategy {
	case algorithm.StrategyAVX512, algorithm.StrategyAVX2:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := algorithm.AdaptiveCosineSimilarity(a, b, strategy)
			return 1.0 - sim, nil
		})
	default:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := algorithm.CosineSimilarity(a, b)
			return 1.0 - sim, nil
		})
	}

	// 执行搜索
	normalizedQuery := algorithm.NormalizeVector(query)
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
			log.Warning("PQ粗排失败，回退到IVF: %v", err)
			candidates, err = db.ivfSearchWithScores(query, coarseK, ctx.Nprobe, db.GetSelectStrategy(query))
		}
	} else {
		candidates, err = db.ivfSearchWithScores(query, coarseK, ctx.Nprobe, db.GetSelectStrategy(query))
	}

	if err != nil {
		return nil, err
	}

	// 第二阶段：精确重排
	candidateIDs := make([]string, len(candidates))
	for i, candidate := range candidates {
		candidateIDs[i] = candidate.Id
	}

	// 使用GPU或SIMD加速精排
	strategy := db.strategyComputeSelector.SelectOptimalStrategy(len(candidateIDs), len(query))
	finalResults, err := db.fineRankingWithScores(query, candidateIDs, k, strategy)
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
	log.Trace("性能指标更新 - 策略: %v, 延迟: %v, 结果数: %d, 总查询数: %d, 平均延迟: %v, 内存使用: %d bytes",
		strategy, latency, resultCount, db.stats.TotalQueries, db.stats.AvgQueryTime, db.stats.MemoryUsage)

	// 定期输出性能摘要（每1000次查询）
	if db.stats.TotalQueries%1000 == 0 {
		db.logPerformanceSummary()
	}
}

// logPerformanceSummary 输出性能摘要
func (db *VectorDB) logPerformanceSummary() {
	log.Info("=== 性能摘要 (查询数: %d) ===", db.stats.TotalQueries)
	log.Info("平均查询时间: %v", db.stats.AvgQueryTime)
	log.Info("内存使用: %.2f MB", float64(db.stats.MemoryUsage)/1024/1024)
	log.Info("缓存命中率: %.2f%%", float64(db.stats.CacheHits)*100/float64(db.stats.TotalQueries))

	// 输出各策略的性能表现
	if db.strategySelector != nil && db.strategySelector.performance != nil {
		log.Info("各策略性能表现:")
		for strategy, metrics := range db.strategySelector.performance {
			strategyName := db.getStrategyName(strategy)
			log.Info("  %s: 延迟=%v, QPS=%.2f, 召回率=%.2f%%, 内存=%d bytes",
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

	for strategy, metrics := range db.strategySelector.performance {
		// 综合评分：考虑延迟、QPS和召回率
		// 分数 = (召回率 * QPS) / 延迟(秒)
		latencySeconds := metrics.AvgLatency.Seconds()
		if latencySeconds > 0 {
			score := (metrics.Recall * metrics.ThroughputQPS) / latencySeconds
			if score > bestScore {
				bestScore = score
				bestStrategy = strategy
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
				log.Warning("构建HNSW索引失败: %v", err)
			}
		}

		if !db.usePQCompression && db.pqCodebook != nil {
			db.usePQCompression = true
			if err := db.CompressExistingVectors(); err != nil {
				log.Warning("PQ压缩失败: %v", err)
			}
		}
	} else if vectorCount > 100000 {
		// 中等规模数据，选择性启用优化
		if !db.useHNSWIndex {
			db.useHNSWIndex = true
			if err := db.BuildHNSWIndex(); err != nil {
				log.Warning("构建HNSW索引失败: %v", err)
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

	log.Info("自适应重建索引完成，数据量: %d", vectorCount)
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

	log.Info("Closing VectorDB...")
	// 发送停止信号给后台任务
	if db.stopCh != nil {
		close(db.stopCh) // 关闭channel以通知goroutine停止
		log.Info("Stop signal sent to background tasks.")
	}
	// 尝试保存数据到文件
	if db.backupPath != "" {
		log.Info("Attempting to save VectorDB data to %s before closing...", db.filePath)
		if err := db.SaveToFile(db.backupPath); err != nil {
			log.Error("Error saving VectorDB data to %s: %v", db.filePath, err)
		}
	}

	// 清理内存中的数据结构
	db.vectors = make(map[string][]float64) // 清空向量
	db.clusters = make([]Cluster, 0)        // 清空簇信息
	db.indexed = false                      // 重置索引状态

	db.invertedMu.Lock()
	db.invertedIndex = make(map[string][]string) // 清空倒排索引
	db.invertedMu.Unlock()

	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache) // 清空查询缓存
	db.cacheMu.Unlock()

	db.normalizedVectors = make(map[string][]float64)               // 清空归一化向量
	db.compressedVectors = make(map[string]entity.CompressedVector) // 清空压缩向量

	// 重置其他可能的状态字段
	db.vectorDim = 0

	log.Info("VectorDB closed successfully.")
}

func (db *VectorDB) IsIndexed() bool {
	return db.indexed
}

// NewVectorDB 创建一个新的 VectorDB 实例。
// 如果 filePath 非空且文件存在，则尝试从中加载数据。
// numClusters 指定了用于索引的簇数量，如果 <=0，则不启用索引功能。
func NewVectorDB(filePath string, numClusters int) *VectorDB {
	db := &VectorDB{
		vectors:           make(map[string][]float64),
		filePath:          filePath,
		numClusters:       numClusters,
		clusters:          make([]Cluster, 0),
		indexed:           false,
		invertedIndex:     make(map[string][]string),
		queryCache:        make(map[string]queryCache),
		cacheTTL:          300, // 默认缓存5分钟
		vectorDim:         0,
		vectorizedType:    DefaultVectorized,
		normalizedVectors: make(map[string][]float64),
		config:            AdaptiveConfig{},

		// 初始化 PQ 相关字段
		pqCodebook:               nil,
		numSubvectors:            0, // 默认为0，表示未配置或不使用
		numCentroidsPerSubvector: 0, // 默认为0
		usePQCompression:         false,
		stopCh:                   make(chan struct{}), // 初始化stopCh

		useHNSWIndex:   false,
		maxConnections: 16,    // 默认值
		efConstruction: 100.0, // 默认值
		efSearch:       50.0,  // 默认值

		// 初始化硬件自适应组件
		strategyComputeSelector: algorithm.NewComputeStrategySelector(),
		currentStrategy:         algorithm.StrategyStandard,
		strategySelector:        &StrategySelector{},
	}
	// 检测硬件能力
	db.hardwareCaps = db.strategyComputeSelector.GetHardwareCapabilities()
	log.Info("硬件检测结果: AVX2=%v, AVX512=%v, GPU=%v, CPU核心=%d",
		db.hardwareCaps.HasAVX2, db.hardwareCaps.HasAVX512,
		db.hardwareCaps.HasGPU, db.hardwareCaps.CPUCores)

	// 如果支持GPU，初始化GPU加速器
	if db.hardwareCaps.HasGPU {
		db.gpuAccelerator = algorithm.NewFAISSGPUAccelerator(0, "Flat")
		if err := db.gpuAccelerator.Initialize(); err != nil {
			log.Warning("GPU加速器初始化失败: %v", err)
			db.hardwareCaps.HasGPU = false
		}
	}
	if filePath != "" {
		if err := db.LoadFromFile(filePath); err != nil {
			log.Warning("警告: 从 %s 加载向量数据库时出错: %v。将使用空数据库启动。\n", filePath, err)
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
			db.normalizedVectors = make(map[string][]float64)
			db.compressedVectors = make(map[string]entity.CompressedVector) // 确保加载失败也初始化
		}
		db.backupPath = filePath + ".bat"
	}
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
	strategy := db.strategyComputeSelector.SelectOptimalStrategy(dataSize, vectorDim)

	log.Trace("选择计算策略: %v, 数据量: %d, 向量维度: %d", strategy, dataSize, vectorDim)

	switch strategy {
	case algorithm.StrategyGPU:
		return db.gpuBatchCosineSimilarity(queries, targets)
	case algorithm.StrategyAVX512, algorithm.StrategyAVX2:
		return db.simdBatchCosineSimilarity(queries, targets, strategy)
	default:
		return db.standardBatchCosineSimilarity(queries, targets)
	}
}

// gpuBatchCosineSimilarity GPU批量余弦相似度计算
func (db *VectorDB) gpuBatchCosineSimilarity(queries [][]float64, targets [][]float64) ([][]float64, error) {
	if db.gpuAccelerator == nil {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	results, err := db.gpuAccelerator.BatchCosineSimilarity(queries, targets)
	if err != nil {
		// GPU计算失败，回退到CPU
		log.Warning("GPU计算失败，回退到CPU: %v", err)
		return db.standardBatchCosineSimilarity(queries, targets)
	}

	return results, nil
}

// simdBatchCosineSimilarity SIMD批量余弦相似度计算
func (db *VectorDB) simdBatchCosineSimilarity(queries [][]float64, targets [][]float64, strategy algorithm.ComputeStrategy) ([][]float64, error) {
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
					similarities[targetIdx] = algorithm.AdaptiveCosineSimilarity(query, target, strategy)
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
					similarities[targetIdx] = algorithm.CosineSimilarity(query, target)
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
func (db *VectorDB) AdaptiveFindNearest(query []float64, k int, nprobe int) ([]string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return []string{}, nil
	}

	// 选择最优计算策略
	dataSize := len(db.vectors)
	vectorDim := len(query)
	strategy := db.strategyComputeSelector.SelectOptimalStrategy(dataSize, vectorDim)

	log.Trace("自适应搜索策略: %v", strategy)

	// 如果启用了HNSW索引，优先使用
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		return db.hnswAdaptiveSearch(query, k, strategy)
	}

	// 使用IVF索引进行自适应搜索
	return db.ivfAdaptiveSearch(query, k, nprobe, strategy)
}

// hnswAdaptiveSearch HNSW自适应搜索
func (db *VectorDB) hnswAdaptiveSearch(query []float64, k int, strategy algorithm.ComputeStrategy) ([]string, error) {
	// 根据策略设置距离函数
	switch strategy {
	case algorithm.StrategyAVX512, algorithm.StrategyAVX2:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := algorithm.AdaptiveCosineSimilarity(a, b, strategy)
			return 1.0 - sim, nil
		})
	default:
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := algorithm.CosineSimilarity(a, b)
			return 1.0 - sim, nil
		})
	}

	// 执行搜索
	normalizedQuery := algorithm.NormalizeVector(query)
	results, err := db.hnsw.Search(normalizedQuery, k)
	if err != nil {
		return nil, err
	}

	// 提取ID
	ids := make([]string, len(results))
	for i, result := range results {
		ids[i] = result.Id
	}

	return ids, nil
}

// ivfAdaptiveSearch IVF自适应搜索
func (db *VectorDB) ivfAdaptiveSearch(query []float64, k int, nprobe int, strategy algorithm.ComputeStrategy) ([]string, error) {
	if !db.indexed {
		return nil, fmt.Errorf("数据库尚未建立索引")
	}

	// 粗排：找到最近的nprobe个簇
	candidateClusters := make([]int, 0, nprobe)
	clusterDistances := make([]float64, len(db.clusters))

	for i, cluster := range db.clusters {
		// 使用自适应距离计算
		switch strategy {
		case algorithm.StrategyAVX512, algorithm.StrategyAVX2:
			sim := algorithm.AdaptiveCosineSimilarity(query, cluster.Centroid, strategy)
			clusterDistances[i] = 1.0 - sim
		default:
			sim := algorithm.CosineSimilarity(query, cluster.Centroid)
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
func (db *VectorDB) adaptiveFineRanking(query []float64, candidates []string, k int, strategy algorithm.ComputeStrategy) ([]string, error) {
	if len(candidates) == 0 {
		return []string{}, nil
	}

	// 根据策略选择计算方法
	switch strategy {
	case algorithm.StrategyGPU:
		return db.gpuFineRanking(query, candidates, k)
	default:
		return db.cpuFineRanking(query, candidates, k, strategy)
	}
}

// cpuFineRanking CPU精排（支持SIMD加速）
func (db *VectorDB) cpuFineRanking(query []float64, candidates []string, k int, strategy algorithm.ComputeStrategy) ([]string, error) {
	type result struct {
		id         string
		similarity float64
	}

	results := make([]result, 0, len(candidates))
	resultsChan := make(chan result, len(candidates))

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
					sim := algorithm.AdaptiveCosineSimilarity(query, vec, strategy)
					resultsChan <- result{id: candidateID, similarity: sim}
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
		return results[i].similarity > results[j].similarity
	})

	if k > len(results) {
		k = len(results)
	}

	topKIDs := make([]string, k)
	for i := 0; i < k; i++ {
		topKIDs[i] = results[i].id
	}

	return topKIDs, nil
}

// gpuFineRanking GPU精排
func (db *VectorDB) gpuFineRanking(query []float64, candidates []string, k int) ([]string, error) {
	if db.gpuAccelerator == nil {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	// 准备候选向量数据
	candidateVectors := make([][]float64, 0, len(candidates))
	validCandidates := make([]string, 0, len(candidates))

	for _, candidateID := range candidates {
		if vec, exists := db.vectors[candidateID]; exists {
			candidateVectors = append(candidateVectors, vec)
			validCandidates = append(validCandidates, candidateID)
		}
	}

	if len(candidateVectors) == 0 {
		return []string{}, nil
	}

	// 使用GPU计算相似度
	queries := [][]float64{query}
	similarities, err := db.gpuAccelerator.BatchCosineSimilarity(queries, candidateVectors)
	if err != nil {
		// GPU计算失败，回退到CPU
		log.Warning("GPU精排失败，回退到CPU: %v", err)
		return db.cpuFineRanking(query, candidates, k, algorithm.StrategyStandard)
	}

	// 处理结果
	type result struct {
		id         string
		similarity float64
	}

	results := make([]result, len(validCandidates))
	for i, candidateID := range validCandidates {
		results[i] = result{
			id:         candidateID,
			similarity: similarities[0][i],
		}
	}

	// 排序并返回top-k
	sort.Slice(results, func(i, j int) bool {
		return results[i].similarity > results[j].similarity
	})

	if k > len(results) {
		k = len(results)
	}

	topKIDs := make([]string, k)
	for i := 0; i < k; i++ {
		topKIDs[i] = results[i].id
	}

	return topKIDs, nil
}

// SetComputeStrategy 手动设置计算策略
func (db *VectorDB) SetComputeStrategy(strategy algorithm.ComputeStrategy) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 验证策略是否可用
	switch strategy {
	case algorithm.StrategyAVX2:
		if !db.hardwareCaps.HasAVX2 {
			return fmt.Errorf("当前硬件不支持AVX2指令集")
		}
	case algorithm.StrategyAVX512:
		if !db.hardwareCaps.HasAVX512 {
			return fmt.Errorf("当前硬件不支持AVX512指令集")
		}
	case algorithm.StrategyGPU:
		if !db.hardwareCaps.HasGPU {
			return fmt.Errorf("当前系统不支持GPU加速")
		}
	}

	db.currentStrategy = strategy
	log.Info("手动设置计算策略为: %v", strategy)
	return nil
}

// GetHardwareInfo 获取硬件信息
func (db *VectorDB) GetHardwareInfo() algorithm.HardwareCapabilities {
	return db.hardwareCaps
}

// GetCurrentStrategy 获取当前计算策略
func (db *VectorDB) GetCurrentStrategy() algorithm.ComputeStrategy {
	return db.currentStrategy
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
			processedVectors[i] = algorithm.NormalizeVector(vec)
		} else {
			processedVectors[i] = vec
		}
	}

	// 使用 HNSW 图的并行添加节点方法
	startTime := time.Now()
	log.Info("开始批量添加 %d 个向量到 HNSW 索引...", len(ids))

	err := db.hnsw.ParallelAddNodes(ids, processedVectors, numWorkers)
	if err != nil {
		return fmt.Errorf("批量添加节点失败: %w", err)
	}

	log.Info("成功批量添加 %d 个向量到 HNSW 索引，耗时 %v", len(ids), time.Since(startTime))
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
	log.Info("开始并行构建 HNSW 索引...")

	// 重置索引状态
	db.indexed = false

	// 创建新的 HNSW 图
	db.hnsw = graph.NewHNSWGraph(db.maxConnections, db.efConstruction, db.efSearch)

	// 设置距离函数
	db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
		// 使用余弦距离（1 - 余弦相似度）
		sim := algorithm.CosineSimilarity(a, b)
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

	log.Info("HNSW 索引并行构建完成，耗时 %v，包含 %d 个向量。", db.stats.IndexBuildTime, len(db.vectors))
	return nil
}

// BatchFindNearest 批量查找多个查询向量的最近邻
// queryVectors: 查询向量数组
// k: 每个查询返回的最近邻数量
// numWorkers: 并行工作的协程数量，如果 <= 0，则使用 CPU 核心数
// 返回: 每个查询向量对应的最近邻ID数组，以及可能的错误
func (db *VectorDB) BatchFindNearest(queryVectors [][]float64, k int, numWorkers int) ([][]string, error) {
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
		emptyResults := make([][]string, len(queryVectors))
		for i := range emptyResults {
			emptyResults[i] = []string{}
		}
		return emptyResults, nil
	}

	// 如果启用了 HNSW 索引，使用 HNSW 批量搜索
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		log.Trace("使用 HNSW 索引进行批量搜索，查询数量: %d", len(queryVectors))

		// 预处理查询向量（归一化）
		normalizedQueries := make([][]float64, len(queryVectors))
		for i, query := range queryVectors {
			normalizedQueries[i] = algorithm.NormalizeVector(query)
		}

		// 使用 HNSW 批量搜索
		batchResults, err := db.hnsw.BatchSearch(normalizedQueries, k, numWorkers)
		if err != nil {
			return nil, fmt.Errorf("HNSW 批量搜索失败: %w", err)
		}

		// 提取 ID
		results := make([][]string, len(batchResults))
		for i, queryResult := range batchResults {
			ids := make([]string, len(queryResult))
			for j, result := range queryResult {
				ids[j] = result.Id
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
	results := make([][]string, len(queryVectors))
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
		log.Trace("使用 HNSW 索引进行批量搜索（带分数），查询数量: %d", len(queryVectors))

		// 预处理查询向量（归一化）
		normalizedQueries := make([][]float64, len(queryVectors))
		for i, query := range queryVectors {
			normalizedQueries[i] = algorithm.NormalizeVector(query)
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
		log.Info("启用 HNSW 索引，需要重建索引。请调用 BuildIndex() 方法。")
	}
}

// BuildHNSWIndex 构建 HNSW 图结构索引
func (db *VectorDB) BuildHNSWIndex() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	startTime := time.Now()
	log.Info("开始构建 HNSW 索引...")

	// 重置索引状态
	db.indexed = false

	// 创建新的 HNSW 图
	db.hnsw = graph.NewHNSWGraph(db.maxConnections, db.efConstruction, db.efSearch)

	// 设置距离函数
	db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
		// 使用余弦距离（1 - 余弦相似度）
		sim := algorithm.CosineSimilarity(a, b)
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

	log.Info("HNSW 索引构建完成，耗时 %v，包含 %d 个向量。", db.stats.IndexBuildTime, len(db.vectors))
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
			log.Info("Vector dimension was not initialized, inferred as %d from existing vectors.", db.vectorDim)
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

	log.Info("从 VectorDB 采样了 %d 个向量用于训练", len(sampledVectors))
	return sampledVectors, nil
}

// LoadPQCodebookFromFile 从文件加载 PQ 码本
func (db *VectorDB) LoadPQCodebookFromFile(filePath string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if filePath == "" {
		log.Warning("PQ 码本文件路径为空，跳过加载。")
		db.pqCodebook = nil
		db.usePQCompression = false // 如果码本路径为空，则禁用PQ
		return nil
	}

	file, err := os.Open(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Warning("PQ 码本文件 %s 不存在，PQ 压缩将不可用。", filePath)
			db.pqCodebook = nil
			db.usePQCompression = false
			return nil // 文件不存在不是致命错误，只是PQ不可用
		}
		return fmt.Errorf("打开 PQ 码本文件 %s 失败: %v", filePath, err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var codebook [][]entity.Point
	if err := decoder.Decode(&codebook); err != nil {
		return fmt.Errorf("解码 PQ 码本文件 %s 失败: %v", filePath, err)
	}

	db.pqCodebook = codebook
	db.pqCodebookFilePath = filePath // 存储路径以备将来热更新检查
	// 可以在这里根据码本结构验证 numSubvectors 和 numCentroidsPerSubvector
	if len(codebook) > 0 {
		db.numSubvectors = len(codebook)
		if len(codebook[0]) > 0 {
			db.numCentroidsPerSubvector = len(codebook[0])
		} else {
			log.Warning("加载的 PQ 码本子空间为空，PQ 参数可能不正确。")
			db.numCentroidsPerSubvector = 0
		}
	} else {
		log.Warning("加载的 PQ 码本为空，PQ 参数可能不正确。")
		db.numSubvectors = 0
		db.numCentroidsPerSubvector = 0
	}

	log.Info("成功从 %s 加载 PQ 码本。子空间数: %d, 每子空间质心数: %d", filePath, db.numSubvectors, db.numCentroidsPerSubvector)
	return nil
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
func (db *VectorDB) SearchWithFilter(query string, topK int, filter func(map[string]interface{}) bool) ([]SearchResult, error) {
	// 将查询文本向量化
	queryVector, err := db.GetVectorForText(query, db.vectorizedType)
	if err != nil {
		return nil, fmt.Errorf("查询文本向量化失败: %v", err)
	}

	// 查找最近的向量
	results, err := db.FindNearest(queryVector, topK*2, 10) // 获取更多结果用于过滤
	if err != nil {
		return nil, fmt.Errorf("搜索失败: %v", err)
	}

	// 应用过滤器并计算相似度
	var filteredResults []SearchResult
	for _, id := range results {
		// 获取元数据
		metadata, exists := db.GetMetadata(id)
		if !exists {
			metadata = make(map[string]interface{})
		}

		// 应用过滤器
		if filter == nil || filter(metadata) {
			// 计算相似度
			similarity, err := db.CalculateCosineSimilarity(id, queryVector)
			if err != nil {
				continue
			}

			filteredResults = append(filteredResults, SearchResult{
				ID:         id,
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
		normalizedQuery = algorithm.NormalizeVector(queryVector)
	}

	// 计算余弦相似度
	similarity := algorithm.CosineSimilarity(normalizedQuery, vector)
	if similarity < 0 {
		return 0, fmt.Errorf("计算余弦相似度失败：向量维度不匹配")
	}

	return similarity, nil
}

// SearchResult 搜索结果结构体
type SearchResult struct {
	ID         string
	Similarity float64
	Metadata   map[string]interface{}
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
		log.Warning("未提供 PQ 码本文件路径，且之前未配置，PQ 压缩将禁用。")
		db.usePQCompression = false
		db.pqCodebook = nil
		db.numSubvectors = 0
		db.numCentroidsPerSubvector = 0
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
		db.numSubvectors = 0
		db.numCentroidsPerSubvector = 0
		db.mu.Unlock()
		return fmt.Errorf("启用 PQ 压缩失败，加载码本时出错: %v", err)
	}
	db.mu.Lock() // 确保在更新 usePQCompression 之前获取锁
	// 只有当码本成功加载且非空时，才真正启用PQ
	if db.pqCodebook != nil && len(db.pqCodebook) > 0 {
		db.usePQCompression = true
		// 更新 numSubVectors 和 numCentroidsPerSubVector 以匹配加载的码本
		db.numSubvectors = len(db.pqCodebook)
		if len(db.pqCodebook[0]) > 0 {
			db.numCentroidsPerSubvector = len(db.pqCodebook[0])
		} else {
			db.numCentroidsPerSubvector = 0 // 或者报错，取决于策略
		}
		log.Info("PQ 压缩已启用。码本路径: %s, 子向量数: %d, 每子空间质心数: %d", db.pqCodebookFilePath, db.numSubvectors, db.numCentroidsPerSubvector)

		// 提示用户可能需要压缩现有向量
		if len(db.vectors) > 0 && len(db.compressedVectors) < len(db.vectors) {
			err := db.CompressExistingVectors()
			if err != nil {
				return err
			}
		}
	} else {
		db.usePQCompression = false
		log.Warning("PQ 码本加载后为空或加载失败，PQ 压缩已禁用。")
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

	log.Info("开始压缩现有向量...")
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
		log.Info("没有需要压缩的向量。")
		return nil
	}

	log.Info("发现 %d 个未压缩向量，开始批量压缩处理...", totalVectors)

	// 使用批量压缩函数
	numWorkers := runtime.NumCPU() // 使用所有可用CPU核心
	log.Info("使用 %d 个工作协程进行并行压缩", numWorkers)

	startTime := time.Now()
	compressedVectors, err := algorithm.BatchCompressByPQ(
		vectorsToCompress,
		db.pqCodebook,
		db.numSubvectors,
		db.numCentroidsPerSubvector,
		numWorkers,
	)

	if err != nil {
		log.Error("批量压缩向量失败: %v", err)
		return fmt.Errorf("批量压缩向量失败: %w", err)
	}

	// 将压缩结果存储到数据库
	for i, id := range idsToCompress {
		db.compressedVectors[id] = compressedVectors[i]
	}

	elapsedTime := time.Since(startTime)
	log.Info("现有向量批量压缩完成，共压缩了 %d 个向量，耗时 %v，平均每个向量 %.2f 毫秒。",
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
	db.normalizedVectors[id] = algorithm.NormalizeVector(vector)
}

// AddDocument 添加文档并将其转换为向量后存入数据库
func (db *VectorDB) AddDocument(id string, doc string, vectorizedType int) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	vector, err := db.GetVectorForText(doc, vectorizedType) // Use GetVectorForText internally
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
	db.normalizedVectors[id] = algorithm.NormalizeVector(vector)
	// 如果启用了 PQ 压缩，则压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := algorithm.OptimizedCompressByPQ(vector, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector)
		if err != nil {
			// 即使压缩失败，原始向量也已添加，这里只记录错误
			log.Error("为文档 %s 添加时压缩向量失败: %v", id, err)
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

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 添加新文档向量后，索引已失效，请重新调用 BuildIndex()。")
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

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	// 设置索引状态为未索引
	db.indexed = false
	// 使用默认参数重建索引
	// 最大迭代次数设为100，收敛容差设为0.001
	err := db.BuildIndex(100, 0.001)

	// 更新性能统计信息
	if err != nil {
		log.Error("索引重建失败: %v", err)
	}
	db.statsMu.Lock()
	db.stats.IndexBuildTime = time.Since(start)
	db.stats.LastReindexTime = time.Now()

	// 更新内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	db.stats.MemoryUsage = m.Alloc
	db.statsMu.Unlock()
	log.Info("索引重建完成，耗时: %v", time.Since(start))

	return err
}

// GetFilePath 获取数据库文件路径
func (db *VectorDB) GetFilePath() string {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.filePath
}

// SetFilePath 设置数据库文件路径
func (db *VectorDB) SetFilePath(path string) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.filePath = path
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

	// 设置向量维度（如果尚未设置）
	if db.vectorDim == 0 && len(vector) > 0 {
		db.vectorDim = len(vector)
	} else if len(vector) != db.vectorDim && db.vectorDim > 0 {
		log.Fatal("向量维度不匹配: 期望 %d, 实际 %d", db.vectorDim, len(vector))
	}

	db.vectors[id] = vector
	// 预计算并存储归一化向量
	db.normalizedVectors[id] = algorithm.NormalizeVector(vector)
	// 如果启用了 PQ 压缩，则压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := algorithm.OptimizedCompressByPQ(vector, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector)
		if err != nil {
			log.Error("向量 %s 压缩失败: %v。该向量将只以原始形式存储。", id, err)
			// 根据策略，可以选择是否回滚添加操作或仅记录错误
		} else {
			if db.compressedVectors == nil {
				db.compressedVectors = make(map[string]entity.CompressedVector)
			}
			db.compressedVectors[id] = compressedVec
			log.Trace("向量 %s 已压缩并存储。", id)
		}
	}

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 添加新向量后，索引已失效，请重新调用 BuildIndex()。")
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
			log.Warning("增量添加向量 %s 到 HNSW 图失败: %v，索引可能不一致。", id, err)
		}
	} else if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 添加新向量后，索引已失效，请重新调用 BuildIndex()。")
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

	// 清除查询缓存，因为数据已更改
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	log.Info("Vector with id %s deleted successfully.", id)
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
	minDist := math.MaxFloat64
	nearestClusterIndex := -1

	// 使用归一化向量进行距离计算（如果可用）
	var queryVecForDist []float64
	if db.normalizedVectors[id] != nil {
		queryVecForDist = db.normalizedVectors[id]
	} else {
		queryVecForDist = algorithm.NormalizeVector(vector) // 如果没有预计算，则动态计算
	}

	for i, cluster := range db.clusters {
		// 假设簇中心也是归一化的，或者在KMeans时已处理
		dist, err := algorithm.EuclideanDistanceSquared(queryVecForDist, cluster.Centroid) // 或者使用余弦相似度
		if err != nil {
			log.Warning("计算到簇 %d 中心的距离失败: %v", i, err)
			continue
		}
		if dist < minDist {
			minDist = dist
			nearestClusterIndex = i
		}
	}

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
			log.Info("向量 %s 已增量添加到簇 %d。", id, nearestClusterIndex)
		}
	} else {
		log.Warning("未能为向量 %s 找到最近的簇进行增量更新。", id)
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
			log.Warning("重新计算簇 %d 中心时，向量 %s 未找到，跳过此向量。", clusterIndex, vecID)
			continue
		}
		if len(vecData) != db.vectorDim {
			log.Warning("向量 %s 的维度 (%d) 与期望维度 (%d) 不符，跳过此向量。", vecID, len(vecData), db.vectorDim)
			continue
		}

		for i, val := range vecData {
			newCentroid[i] += val
		}
		validVectorsCount++
	}

	if validVectorsCount == 0 {
		log.Warning("簇 %d 中没有有效的向量来计算新的中心点，保留旧中心点。", clusterIndex)
		return nil
	}

	for i := 0; i < db.vectorDim; i++ {
		newCentroid[i] /= float64(len(cluster.VectorIDs))
	}

	cluster.Centroid = newCentroid
	log.Info("簇 %d 的中心点已重新计算。", clusterIndex)
	return nil
}

// StartClusterCentroidUpdater 启动一个定时器，定期更新所有簇的中心点
// updateInterval: 更新间隔，例如 time.Minute * 5 表示每5分钟更新一次
func (db *VectorDB) StartClusterCentroidUpdater(interval time.Duration) {
	if !db.indexed || db.numClusters <= 0 {
		log.Info("索引未启用或簇数量未设置，不启动簇中心更新器。")
		return
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		log.Info("簇中心定期更新器已启动，更新间隔: %s", interval.String())

		for {
			select {
			case <-ticker.C:
				log.Info("开始定期重新计算所有簇中心...")
				db.mu.Lock() // 获取写锁以更新簇中心
				if len(db.clusters) == 0 {
					db.mu.Unlock()
					log.Info("当前没有簇，跳过簇中心重新计算。")
					continue
				}
				for i := range db.clusters {
					if err := db.recalculateClusterCentroid(i); err != nil {
						log.Error("重新计算簇 %d 的中心失败: %v", i, err)
					}
				}
				db.mu.Unlock()
				log.Info("所有簇中心重新计算完成。")
			case <-db.stopCh: // 监听停止信号
				log.Info("接收到停止信号，簇中心更新器正在关闭...")
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
	db.normalizedVectors[id] = algorithm.NormalizeVector(vector)
	// 如果启用了 PQ 压缩，则更新压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := algorithm.OptimizedCompressByPQ(vector, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector)
		if err != nil {
			log.Error("为向量 %s 更新时压缩向量失败: %v", id, err)
			// 即使压缩失败，原始向量也已更新
			delete(db.compressedVectors, id) // 删除旧地压缩向量，因为它不再有效
		} else {
			db.compressedVectors[id] = compressedVec
		}
	}
	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 更新向量后，索引已失效，请重新调用 BuildIndex()。")
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

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

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
		log.Info("提示: 删除向量后，索引已失效，请重新调用 BuildIndex()。")
	}

	// 如果启用了 HNSW 索引，从 HNSW 图中删除节点
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		err := db.hnsw.DeleteNode(id)
		if err != nil {
			log.Warning("从 HNSW 图中删除向量 %s 失败: %v，索引可能不一致。", id, err)
		}
	} else if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 删除向量后，索引已失效，请重新调用 BuildIndex()。")
	}

	return nil
}

// MultiLevelIndex 多级索引结构
type MultiLevelIndex struct {
	// 一级索引：簇中心
	clusters []Cluster

	// 二级索引：每个簇内部的KD树或其他数据结构
	subIndices []interface{} // 可以是KDTree或其他索引结构

	// 索引元数据
	numClusters int
	indexed     bool
	buildTime   time.Time
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
		clusters:    db.clusters,                         // 注意：这里可能需要深拷贝或调整，取决于 MultiLevelIndex 的设计
		subIndices:  make([]interface{}, db.numClusters), // 假设 subIndices 在 goroutine 中填充
		numClusters: db.numClusters,
		indexed:     true,
		buildTime:   time.Now(),
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
			multiIndex.subIndices[clusterIdx] = kdTree
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

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

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

	log.Info("开始构建索引...")
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
			log.Warning("警告: 向量 %s 被分配到无效的簇索引 %d\n", vectorIDs[i], clusterIndex)
		}
	}

	db.indexed = true
	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	log.Info("索引构建完成，共 %d 个簇。\n", db.numClusters)
	return nil
}

// dataToSave 结构用于 gob 编码，包含所有需要持久化的字段
type dataToSave struct {
	Vectors                  map[string][]float64
	Clusters                 []Cluster
	NumClusters              int
	Indexed                  bool
	InvertedIndex            map[string][]string
	VectorDim                int
	NormalizedVectors        map[string][]float64
	CompressedVectors        map[string]entity.CompressedVector
	PQCodebook               []float64
	NumSubvectors            int
	NumCentroidsPerSubvector int
	UsePQCompression         bool
}

// SaveToFile 将当前数据库状态（包括索引）保存到其配置的文件中。
func (db *VectorDB) SaveToFile(filePath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.backupPath == "" {
		return fmt.Errorf("文件路径未设置，无法保存数据库")
	}

	file, err := os.Create(db.backupPath)
	if err != nil {
		return fmt.Errorf("创建数据库文件 %s 失败: %v", db.backupPath, err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Error("close file failed: %v", err)
		}
	}(file)

	encoder := gob.NewEncoder(file)

	// 序列化 VectorDB 的核心数据
	// 为了向前兼容和模块化，可以考虑为每个主要部分创建独立的结构进行序列化
	data := struct {
		Vectors           map[string][]float64
		Clusters          []Cluster
		NumClusters       int
		Indexed           bool
		InvertedIndex     map[string][]string
		VectorDim         int
		VectorizedType    int
		NormalizedVectors map[string][]float64
		CompressedVectors map[string]entity.CompressedVector
		UseCompression    bool
		// PQ 相关字段也需要保存，以便下次加载时能正确恢复状态
		PqCodebookFilePath       string // 保存码本路径，而不是码本本身，码本由外部文件管理
		NumSubvectors            int
		NumCentroidsPerSubvector int
		UsePQCompression         bool
		MultiIndex               *MultiLevelIndex // 如果 MultiLevelIndex 可序列化
		Config                   AdaptiveConfig   // 如果 AdaptiveConfig 可序列化
		UseHNSWIndex             bool
		MaxConnections           int
		EfConstruction           float64
		EfSearch                 float64
	}{
		Vectors:                  db.vectors,
		Clusters:                 db.clusters,
		NumClusters:              db.numClusters,
		Indexed:                  db.indexed,
		InvertedIndex:            db.invertedIndex,
		VectorDim:                db.vectorDim,
		VectorizedType:           db.vectorizedType,
		NormalizedVectors:        db.normalizedVectors,
		CompressedVectors:        db.compressedVectors,
		UseCompression:           db.useCompression,
		PqCodebookFilePath:       db.pqCodebookFilePath, // 保存码本文件路径
		NumSubvectors:            db.numSubvectors,
		NumCentroidsPerSubvector: db.numCentroidsPerSubvector,
		UsePQCompression:         db.usePQCompression,
		MultiIndex:               db.multiIndex,
		Config:                   db.config,
		UseHNSWIndex:             db.useHNSWIndex,
		MaxConnections:           db.maxConnections,
		EfConstruction:           db.efConstruction,
		EfSearch:                 db.efSearch,
	}

	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("序列化数据库到 %s 失败: %v", db.filePath, err)
	}
	// 如果启用了 HNSW 索引，保存 HNSW 图结构
	if db.useHNSWIndex && db.hnsw != nil {
		hnswFilePath := filePath + ".hnsw"
		err := db.hnsw.SaveToFile(hnswFilePath)
		if err != nil {
			return fmt.Errorf("保存 HNSW 图结构失败: %w", err)
		}
	}
	log.Info("VectorDB 数据成功保存到 %s", db.filePath)

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

// LoadFromFile 从其配置的文件中加载数据库状态（包括索引）。
func (db *VectorDB) LoadFromFile(filePath string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.filePath == "" {
		return fmt.Errorf("文件路径未设置，无法加载数据库")
	}

	file, err := os.Open(db.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Info("数据库文件 %s 不存在，将创建一个新的空数据库。", db.filePath)
			// 初始化为空数据库状态，确保所有 map 都已创建
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
			db.invertedIndex = make(map[string][]string)
			db.queryCache = make(map[string]queryCache)
			db.normalizedVectors = make(map[string][]float64)
			db.compressedVectors = make(map[string]entity.CompressedVector)
			db.pqCodebook = nil
			return nil // 文件不存在不是错误，是正常启动流程
		}
		return fmt.Errorf("打开数据库文件 %s 失败: %v", db.filePath, err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	data := struct {
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
		NumSubvectors            int
		NumCentroidsPerSubvector int
		UsePQCompression         bool

		UseHNSWIndex   bool
		MaxConnections int
		EfConstruction float64
		EfSearch       float64
	}{}

	if err := decoder.Decode(&data); err != nil {
		// 如果解码失败，可能是文件损坏或格式不兼容
		// 记录错误，并以空数据库启动，避免程序崩溃
		log.Error("从 %s 反序列化数据库失败: %v。将使用空数据库启动。", db.filePath, err)
		db.vectors = make(map[string][]float64)
		db.clusters = make([]Cluster, 0)
		db.indexed = false
		db.invertedIndex = make(map[string][]string)
		db.queryCache = make(map[string]queryCache)
		db.normalizedVectors = make(map[string][]float64)
		db.compressedVectors = make(map[string]entity.CompressedVector)
		db.pqCodebook = nil
		return nil // 即使加载失败，也返回nil，让程序继续运行
	}

	// 恢复数据
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
	db.numSubvectors = data.NumSubvectors
	db.numCentroidsPerSubvector = data.NumCentroidsPerSubvector
	db.usePQCompression = data.UsePQCompression
	// 从加载的数据中恢复 HNSW 相关字段
	db.useHNSWIndex = data.UseHNSWIndex
	db.maxConnections = data.MaxConnections
	db.efConstruction = data.EfConstruction
	db.efSearch = data.EfSearch

	// 确保 map 在 nil 的情况下被初始化
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
	if db.queryCache == nil { // queryCache 不在 gob 中，需要单独初始化
		db.queryCache = make(map[string]queryCache)
	}

	// 如果启用了 PQ 压缩且有码本路径，则尝试加载码本
	if db.usePQCompression && db.pqCodebookFilePath != "" {
		// 这里使用临时变量，避免在 LoadPQCodebookFromFile 中发生死锁
		tempPath := db.pqCodebookFilePath
		db.pqCodebookFilePath = "" // 暂时清除，避免 LoadPQCodebookFromFile 内部逻辑冲突
		db.usePQCompression = false

		db.mu.Unlock() // 解锁以便 LoadPQCodebookFromFile 可以获取锁
		errLoadCodebook := db.LoadPQCodebookFromFile(tempPath)
		db.mu.Lock() // 重新获取锁

		if errLoadCodebook != nil {
			log.Error("从 %s 加载数据库后，尝试加载 PQ 码本 %s 失败: %v。PQ 压缩将禁用。", db.filePath, tempPath, errLoadCodebook)
			db.usePQCompression = false
			db.pqCodebook = nil
		} else {
			// LoadPQCodebookFromFile 会更新 db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector
			// 它也会在成功加载码本后设置 db.usePQCompression = true (如果码本非空)
			// 所以这里我们只需要确保 db.usePQCompression 反映了加载结果
			if db.pqCodebook == nil || len(db.pqCodebook) == 0 {
				db.usePQCompression = false
			} else {
				db.usePQCompression = true // 确保与加载的码本状态一致
			}
		}
		// 恢复原始配置的 usePQCompression 状态，如果码本加载失败，则它会被设为 false
		// 如果码本加载成功，LoadPQCodebookFromFile 内部会处理
		// 实际上，我们应该信任 LoadPQCodebookFromFile 设置的 usePQCompression
		// 所以，如果 tempUsePQ 为 true 但加载失败，usePQCompression 会是 false，这是正确的
	} else if db.usePQCompression && db.pqCodebookFilePath == "" {
		log.Warning("数据库配置为使用 PQ 压缩，但未指定码本文件路径。PQ 压缩将禁用。")
		db.usePQCompression = false
		db.pqCodebook = nil
	}
	// 如果启用了 HNSW 索引，加载 HNSW 图结构
	if db.useHNSWIndex {
		hnswFilePath := filePath + ".hnsw"
		db.hnsw = graph.NewHNSWGraph(db.maxConnections, db.efConstruction, db.efSearch)
		// 尝试加载 HNSW 图结构
		err := db.hnsw.LoadFromFile(hnswFilePath)

		// 设置距离函数,。在加载后，需要使用 SetDistanceFunc 方法重新设置距离函数
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			// 使用余弦距离（1 - 余弦相似度）
			sim := algorithm.CosineSimilarity(a, b)
			return 1.0 - sim, nil
		})

		if err != nil {
			log.Warning("加载 HNSW 图结构失败: %v，将重新构建索引。", err)
			db.indexed = false
		}
	}
	// 设置备份路径
	db.backupPath = filePath + ".bat"
	log.Info("VectorDB 数据成功从 %s 加载。向量数: %d, 是否已索引: %t, PQ压缩: %t", db.filePath, len(db.vectors), db.indexed, db.usePQCompression)
	return nil
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
	dist, err := algorithm.ApproximateDistanceADC(queryVector, compressedDBVector, db.pqCodebook, db.numSubvectors)
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
				normalized := algorithm.NormalizeVector(item.vec)
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

	log.Info("批量归一化完成，处理了 %d 个向量", len(db.vectors))
	return nil
}

// PCAConfig PCA 配置
type PCAConfig struct {
	TargetDimension int         // 目标维度
	VarianceRatio   float64     // 保留的方差比例
	Components      [][]float64 // PCA 主成分
	Mean            []float64   // 均值向量
}

// ApplyPCA 应用 PCA 降维
func (db *VectorDB) ApplyPCA(targetDim int, varianceRatio float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(db.vectors) == 0 {
		return fmt.Errorf("数据库为空，无法进行 PCA")
	}

	// 收集所有向量
	vectors := make([][]float64, 0, len(db.vectors))
	ids := make([]string, 0, len(db.vectors))

	for id, vec := range db.vectors {
		vectors = append(vectors, vec)
		ids = append(ids, id)
	}

	// 计算均值
	dim := len(vectors[0])
	mean := make([]float64, dim)
	for _, vec := range vectors {
		for i, val := range vec {
			mean[i] += val
		}
	}
	for i := range mean {
		mean[i] /= float64(len(vectors))
	}

	// 中心化数据
	centeredVectors := make([][]float64, len(vectors))
	for i, vec := range vectors {
		centeredVectors[i] = make([]float64, dim)
		for j, val := range vec {
			centeredVectors[i][j] = val - mean[j]
		}
	}

	// 计算协方差矩阵
	covariance := make([][]float64, dim)
	for i := range covariance {
		covariance[i] = make([]float64, dim)
		for j := range covariance[i] {
			for _, vec := range centeredVectors {
				covariance[i][j] += vec[i] * vec[j]
			}
			covariance[i][j] /= float64(len(vectors) - 1)
		}
	}

	// 这里需要实现特征值分解，简化示例
	// 实际应用中建议使用 gonum 等数学库

	// 应用降维变换
	reducedVectors := make(map[string][]float64)
	for i, id := range ids {
		// 简化的降维实现，实际需要使用主成分
		reduced := make([]float64, targetDim)
		for j := 0; j < targetDim && j < len(vectors[i]); j++ {
			reduced[j] = vectors[i][j]
		}
		reducedVectors[id] = reduced
	}

	// 更新向量数据库
	db.vectors = reducedVectors
	db.vectorDim = targetDim
	db.indexed = false // 需要重建索引

	log.Info("PCA 降维完成：%d -> %d 维", dim, targetDim)
	return nil
}

// EnableGPUAcceleration 为 VectorDB 启用 GPU 加速
func (db *VectorDB) EnableGPUAcceleration(gpuID int, indexType string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 创建 GPU 加速器
	db.gpuAccelerator = algorithm.NewFAISSGPUAccelerator(gpuID, indexType)

	// 初始化 GPU 加速器
	err := db.gpuAccelerator.Initialize()
	if err != nil {
		return fmt.Errorf("初始化 GPU 加速器失败: %w", err)
	}

	log.Info("GPU 加速已启用，GPU ID: %d, 索引类型: %s", gpuID, indexType)
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

	log.Trace("粗筛阶段完成，获得 %d 个候选", len(coarseCandidates))

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

	log.Trace("两阶段搜索完成，耗时 %v", queryTime)
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
	normalizedQuery := algorithm.NormalizeVector(query)

	// 找到最近的 nprobe 个簇
	nearestClusters := make([]int, 0, nprobe)
	clusterDists := make([]float64, len(db.clusters))

	for i, cluster := range db.clusters {
		dist, _ := algorithm.EuclideanDistanceSquared(normalizedQuery, cluster.Centroid)
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
		similarity := algorithm.OptimizedCosineSimilarity(query, vec)
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
			cpuResults, cpuErr := db.cpuFineRanking(query, candidates, config.FineK, algorithm.StrategyStandard)
			if cpuErr != nil {
				return nil, cpuErr
			}

			// 将 CPU 结果转换为 entity.Result 格式
			db.appendResults(query, cpuResults, &results, config)
			return results, nil
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
		db.appendResults(query, gpuResults, &results, config)
		return results, nil
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
				similarity := algorithm.OptimizedCosineSimilarity(query, vec)

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
	normalizedQuery := algorithm.NormalizeVector(query)

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
func (db *VectorDB) FindNearest(query []float64, k int, nprobe int) ([]string, error) {
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
//	} else if db.config.UseMultiLevelIndex && db.multiIndex != nil && db.multiIndex.indexed {
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

// AdaptiveNprobeSearch 自适应 nprobe 搜索
func (db *VectorDB) AdaptiveNprobeSearch(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	dataSize := len(db.vectors)

	// 根据数据规模自适应调整 nprobe
	var nprobe int
	switch {
	case dataSize < 10000:
		nprobe = max(1, db.numClusters/4)
	case dataSize < 100000:
		nprobe = max(2, db.numClusters/3)
	case dataSize < 1000000:
		nprobe = max(3, db.numClusters/2)
	default:
		nprobe = max(5, db.numClusters*2/3)
	}

	// 确保 nprobe 在合理范围内
	if nprobe > db.numClusters {
		nprobe = db.numClusters
	}

	return db.ivfSearch(query, k, nprobe)
}

// AdaptiveHNSWConfig HNSW 自适应配置
func (db *VectorDB) AdaptiveHNSWConfig() {
	db.mu.Lock()
	defer db.mu.Unlock()

	dataSize := len(db.vectors)

	// 根据数据规模调整 efConstruction
	switch {
	case dataSize < 10000:
		db.efConstruction = 100.0
	case dataSize < 100000:
		db.efConstruction = 200.0
	case dataSize < 1000000:
		db.efConstruction = 400.0
	default:
		db.efConstruction = 800.0
	}

	// 根据向量维度调整连接数
	if db.vectorDim > 0 {
		db.maxConnections = min(64, max(16, db.vectorDim/10))
	}
}

func (db *VectorDB) GetSelectStrategy(query []float64) algorithm.ComputeStrategy {
	dataSize := len(db.vectors)
	vectorDim := len(query)
	strategy := db.strategyComputeSelector.SelectOptimalStrategy(dataSize, vectorDim)

	return strategy
}

// FindNearestWithScores 查找最近邻并返回分数（更新为使用自适应计算）
func (db *VectorDB) FindNearestWithScores(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return []entity.Result{}, nil
	}

	// 选择最优计算策略
	strategy := db.GetSelectStrategy(query)

	// 如果启用了HNSW索引，优先使用
	if db.useHNSWIndex && db.indexed && db.hnsw != nil {
		// 设置自适应距离函数
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			sim := algorithm.AdaptiveCosineSimilarity(a, b, strategy)
			return 1.0 - sim, nil
		})

		normalizedQuery := algorithm.NormalizeVector(query)
		return db.hnsw.Search(normalizedQuery, k)
	}

	// 使用IVF索引进行自适应搜索
	if !db.indexed {
		return nil, fmt.Errorf("数据库尚未建立索引")
	}

	// IVF搜索逻辑：粗排 + 精排
	return db.ivfSearchWithScores(query, k, nprobe, strategy)
}

// ivfSearchWithScores IVF搜索返回带分数的结果
func (db *VectorDB) ivfSearchWithScores(query []float64, k int, nprobe int, strategy algorithm.ComputeStrategy) ([]entity.Result, error) {
	// 粗排：找到最近的nprobe个簇
	candidateClusters := make([]int, 0, nprobe)
	clusterDistances := make([]float64, len(db.clusters))

	for i, cluster := range db.clusters {
		// 使用自适应距离计算
		switch strategy {
		case algorithm.StrategyAVX512, algorithm.StrategyAVX2:
			sim := algorithm.AdaptiveCosineSimilarity(query, cluster.Centroid, strategy)
			clusterDistances[i] = 1.0 - sim
		default:
			sim := algorithm.CosineSimilarity(query, cluster.Centroid)
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
func (db *VectorDB) fineRankingWithScores(query []float64, candidates []string, k int, strategy algorithm.ComputeStrategy) ([]entity.Result, error) {
	if len(candidates) == 0 {
		return []entity.Result{}, nil
	}

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
					case algorithm.StrategyAVX512, algorithm.StrategyAVX2:
						sim = algorithm.AdaptiveCosineSimilarity(query, vec, strategy)
					default:
						sim = algorithm.CosineSimilarity(query, vec)
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

// FindNearestWithScores_1 查找最近的k个向量，并返回它们的ID和相似度分数
func (db *VectorDB) FindNearestWithScores_1(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	startTime := time.Now()
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}
	if len(db.vectors) == 0 {
		return []entity.Result{}, nil
	}
	normalizedQuery := algorithm.NormalizeVector(query)
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
						sim := algorithm.CosineSimilarity(normalizedQuery, vec)
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
					sim := algorithm.CosineSimilarity(normalizedQuery, vec)
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
	return results, nil
}

// AdaptiveConfig 自适应配置结构
type AdaptiveConfig struct {
	// 索引参数
	NumClusters           int     // 簇数量
	IndexRebuildThreshold float64 // 更新比例阈值，超过此值重建索引

	// 查询参数
	DefaultNprobe int           // 默认探测簇数量
	CacheTimeout  time.Duration // 缓存超时时间

	// 系统参数
	MaxWorkers         int  // 最大工作协程数
	VectorCompression  bool // 是否启用向量压缩
	UseMultiLevelIndex bool // 是否使用多级索引

	// 自适应 nprobe 参数
	MinNprobe    int     // 最小探测簇数
	MaxNprobe    int     // 最大探测簇数
	RecallTarget float64 // 目标召回率

	// HNSW 自适应参数
	MinEfConstruction float64 // 最小构建参数
	MaxEfConstruction float64 // 最大构建参数
	QualityThreshold  float64 // 质量阈值
}

// AdjustConfig 自适应配置调整
func (db *VectorDB) AdjustConfig() {
	db.mu.RLock()
	vectorCount := len(db.vectors)
	db.mu.RUnlock()

	config := db.config

	// 根据向量数量调整簇数量
	if vectorCount > 1000000 {
		config.NumClusters = 1000
	} else if vectorCount > 100000 {
		config.NumClusters = 100
	} else if vectorCount > 10000 {
		config.NumClusters = 50
	} else {
		config.NumClusters = 10
	}

	// 根据系统资源调整工作协程数
	config.MaxWorkers = runtime.NumCPU()

	// 根据内存使用情况决定是否启用向量压缩
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if m.Alloc > 1024*1024*1024 { // 如果内存使用超过1GB
		config.VectorCompression = true
	}

	db.mu.Lock()
	db.config = config
	db.mu.Unlock()
}

// HybridSearch 混合搜索策略
func (db *VectorDB) HybridSearch(query []float64, k int, options SearchOptions, nprobe int) ([]entity.Result, error) {
	// 根据向量维度和数据规模自动选择最佳搜索策略
	if len(db.vectors) < 1000 || !db.indexed {
		// 小数据集使用暴力搜索
		return db.bruteForceSearch(query, k)
	} else if len(query) > 1000 {
		// 高维向量使用LSH (Locality-Sensitive Hashing)
		return db.lshSearch(query, k, options.NumHashTables)
	} else if db.config.UseMultiLevelIndex && db.multiIndex != nil && db.multiIndex.indexed {
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
	normalizedQuery := algorithm.NormalizeVector(query)

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
				sim := algorithm.CosineSimilarity(normalizedQuery, vec)
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
	ids := make([]entity.Result, 0, len(resultHeap))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// LSHTable 局部敏感哈希表结构
type LSHTable struct {
	HashFunctions [][]float64         // 哈希函数参数
	Buckets       map[uint64][]string // 哈希桶，存储向量ID
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
		log.Error("构建LSH索引失败: %v，回退到暴力搜索\n", err)
		return db.bruteForceSearch(query, k)
	}

	// 预处理查询向量
	normalizedQuery := algorithm.NormalizeVector(query)

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
		sim := algorithm.CosineSimilarity(normalizedQuery, vec)

		if len(resultHeap) < k {
			heap.Push(&resultHeap, entity.Result{Id: id, Similarity: sim})
		} else if sim > resultHeap[0].Similarity {
			heap.Pop(&resultHeap)
			heap.Push(&resultHeap, entity.Result{Id: id, Similarity: sim})
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, len(resultHeap))
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
	log.Trace("Using Multi-Level Index for FindNearest.")
	// 1. 找到 nprobe 个最近的簇中心 (与之前逻辑类似)
	clusterDist := make([]struct {
		Index int
		Dist  float64
	}, len(db.multiIndex.clusters))

	for i, cluster := range db.multiIndex.clusters {
		dist, err := algorithm.EuclideanDistanceSquared(query, cluster.Centroid)
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
		selectedCluster := db.multiIndex.clusters[clusterIdx]

		if clusterIdx >= len(db.multiIndex.subIndices) || db.multiIndex.subIndices[clusterIdx] == nil {
			log.Warning("Sub-index for cluster %d not found or nil. Performing brute-force in this cluster.", clusterIdx)
			// 回退到暴力搜索该簇内的向量
			for _, id := range selectedCluster.VectorIDs {
				if vec, exists := db.vectors[id]; exists {
					dist, _ := algorithm.EuclideanDistanceSquared(query, vec)
					heap.Push(&resultHeap, entity.Result{Id: id, Similarity: dist})
					if results.Len() > k {
						heap.Pop(results)
					}
				}
			}
			continue
		}

		// 假设二级索引是 KDTree，并且有 FindNearest 方法
		kdTree, ok := db.multiIndex.subIndices[clusterIdx].(*tree.KDTree) // 类型断言
		if !ok || kdTree == nil {
			log.Warning("Sub-index for cluster %d is not a KDTree or is nil. Performing brute-force.", clusterIdx)
			for _, id := range selectedCluster.VectorIDs {
				if vec, exists := db.vectors[id]; exists {
					dist, _ := algorithm.EuclideanDistanceSquared(query, vec)
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
			log.Error("Error searching in KDTree for cluster %d. Skipping this sub-index.", clusterIdx)
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
	ids := make([]entity.Result, 0, len(resultHeap))
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
		log.Warning("索引未构建，回退到暴力搜索")
		return db.bruteForceSearch(query, k)
	}

	// 预处理查询向量
	normalizedQuery := algorithm.NormalizeVector(query)

	// 设置默认nprobe值
	if nprobe <= 0 {
		nprobe = 1 // 默认搜索最近的一个簇
	}
	if nprobe > db.numClusters {
		nprobe = db.numClusters // 不能超过总簇数
	}

	// 使用堆结构来维护最近的nprobe个簇
	centroidHeap := make(entity.CentroidHeap, 0, db.numClusters)

	// 找到查询向量最近的nprobe个簇中心
	for i, cluster := range db.clusters {
		distSq, err := algorithm.EuclideanDistanceSquared(normalizedQuery, cluster.Centroid)
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
					sim := algorithm.CosineSimilarity(normalizedQuery, vec)
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
	ids := make([]entity.Result, 0, len(resultHeap))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// FileSystemSearch 优化后的FileSystemSearch方法
func (db *VectorDB) FileSystemSearch(query string, vectorizedType int, k int, nprobe int) ([]string, error) {
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
	type result struct {
		id         string
		similarity float64
		wordCount  int
	}

	// 使用工作池并行计算相似度
	numWorkers := runtime.NumCPU()
	workChan := make(chan candidate, len(candidates))
	resultChan := make(chan result, len(candidates))

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
				sim := algorithm.CosineSimilarity(queryVector, vec)
				resultChan <- result{id: c.id, similarity: sim, wordCount: c.count}
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
	results := make([]result, 0, len(candidates))
	for r := range resultChan {
		results = append(results, r)
	}

	// 使用混合排序：先按关键词匹配数量，再按向量相似度
	sort.Slice(results, func(i, j int) bool {
		// 如果关键词匹配数量相差超过阈值，优先考虑匹配数量
		if math.Abs(float64(results[i].wordCount-results[j].wordCount)) > 2 {
			return results[i].wordCount > results[j].wordCount
		}
		// 否则按相似度排序
		return results[i].similarity > results[j].similarity
	})

	// 提取前k个结果
	count := k
	if len(results) < k {
		count = len(results)
	}

	ids := make([]string, count)
	for i := 0; i < count; i++ {
		ids[i] = results[i].id
	}

	return ids, nil
}

// 改进的查询缓存结构
type enhancedQueryCache struct {
	results    []string  // 结果ID列表
	timestamp  time.Time // 缓存创建时间
	vectorHash uint64    // 查询向量的哈希值
}

// ShardedVectorDB 分片锁结构
type ShardedVectorDB struct {
	shards    []*VectorShard
	numShards int
}

type VectorShard struct {
	vectors map[string][]float64
	mu      sync.RWMutex
}

// 根据ID确定分片
func (db *ShardedVectorDB) getShardForID(id string) (*VectorShard, error) {
	h := fnv.New32a()
	_, err := h.Write([]byte(id))
	if err != nil {
		return nil, err
	}
	shardIndex := int(h.Sum32()) % db.numShards
	return db.shards[shardIndex], nil
}

// Get 分片查询实现
func (db *ShardedVectorDB) Get(id string) ([]float64, bool) {
	shard, err := db.getShardForID(id)
	if err != nil {
		return nil, false
	}

	shard.mu.RLock()
	defer shard.mu.RUnlock()
	vec, exists := shard.vectors[id]
	return vec, exists
}

// ParallelFindNearest 优化的并行查询实现
func (db *VectorDB) ParallelFindNearest(query []float64, k int) ([]string, error) {
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
				sim := algorithm.OptimizedCosineSimilarity(query, vec)
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

	// 提取ID
	ids := make([]string, len(allResults))
	for i, result := range allResults {
		ids[i] = result.Id
	}

	return ids, nil
}

func (db *VectorDB) GetDataSize() int {
	return len(db.vectors)
}

// SearchOptions 搜索选项结构
type SearchOptions struct {
	Nprobe        int           // IVF搜索探测的簇数量
	NumHashTables int           // LSH哈希表数量
	UseANN        bool          // 是否使用近似最近邻
	SearchTimeout time.Duration // 搜索超时时间
}

// 多级缓存结构
type MultiLevelCache struct {
	// L1: 内存缓存 - 最快，容量小
	l1Cache    map[string]queryCache
	l1Capacity int
	l1Mu       sync.RWMutex

	// L2: 共享内存缓存 - 较快，容量中等
	l2Cache    map[string]queryCache
	l2Capacity int
	l2Mu       sync.RWMutex

	// L3: 磁盘缓存 - 较慢，容量大
	l3CachePath string
	l3Mu        sync.RWMutex

	// 缓存统计
	stats   CacheStats
	statsMu sync.RWMutex
}

// CacheStats 缓存统计信息
type CacheStats struct {
	L1Hits       int64
	L2Hits       int64
	L3Hits       int64
	TotalQueries int64
}

// NewMultiLevelCache 创建新的多级缓存
func NewMultiLevelCache(l1Capacity, l2Capacity int, l3Path string) *MultiLevelCache {
	return &MultiLevelCache{
		l1Cache:     make(map[string]queryCache, l1Capacity),
		l1Capacity:  l1Capacity,
		l2Cache:     make(map[string]queryCache, l2Capacity),
		l2Capacity:  l2Capacity,
		l3CachePath: l3Path,
		stats:       CacheStats{},
	}
}

// Get 从多级缓存获取结果
func (c *MultiLevelCache) Get(key string) ([]string, bool) {
	// 更新查询计数
	c.statsMu.Lock()
	c.stats.TotalQueries++
	c.statsMu.Unlock()

	// 尝试从 L1 缓存获取
	c.l1Mu.RLock()
	if cache, found := c.l1Cache[key]; found && time.Now().Unix()-cache.timestamp < 300 {
		c.l1Mu.RUnlock()
		c.statsMu.Lock()
		c.stats.L1Hits++
		c.statsMu.Unlock()
		return cache.results, true
	}
	c.l1Mu.RUnlock()

	// 尝试从 L2 缓存获取
	c.l2Mu.RLock()
	if cache, found := c.l2Cache[key]; found && time.Now().Unix()-cache.timestamp < 1800 {
		// 将结果提升到 L1 缓存
		c.l1Mu.Lock()
		c.l1Cache[key] = cache
		// 如果 L1 缓存超出容量，移除最旧的项
		if len(c.l1Cache) > c.l1Capacity {
			var oldestKey string
			var oldestTime int64 = math.MaxInt64
			for k, v := range c.l1Cache {
				if v.timestamp < oldestTime {
					oldestTime = v.timestamp
					oldestKey = k
				}
			}
			delete(c.l1Cache, oldestKey)
		}
		c.l1Mu.Unlock()

		c.l2Mu.RUnlock()
		c.statsMu.Lock()
		c.stats.L2Hits++
		c.statsMu.Unlock()
		return cache.results, true
	}
	c.l2Mu.RUnlock()

	// 尝试从 L3 缓存获取
	// 这里需要实现从磁盘读取缓存的逻辑
	// ...

	return nil, false
}

// Put 将结果存入多级缓存
func (c *MultiLevelCache) Put(key string, results []string) {
	cache := queryCache{
		results:   results,
		timestamp: time.Now().Unix(),
	}

	// 存入 L1 缓存
	c.l1Mu.Lock()
	c.l1Cache[key] = cache
	// 如果 L1 缓存超出容量，移除最旧的项
	if len(c.l1Cache) > c.l1Capacity {
		var oldestKey string
		var oldestTime int64 = math.MaxInt64
		for k, v := range c.l1Cache {
			if v.timestamp < oldestTime {
				oldestTime = v.timestamp
				oldestKey = k
			}
		}
		delete(c.l1Cache, oldestKey)
	}
	c.l1Mu.Unlock()

	// 异步存入 L2 和 L3 缓存
	go func() {
		// 存入 L2 缓存
		c.l2Mu.Lock()
		c.l2Cache[key] = cache
		// 如果 L2 缓存超出容量，移除最旧的项
		if len(c.l2Cache) > c.l2Capacity {
			var oldestKey string
			var oldestTime int64 = math.MaxInt64
			for k, v := range c.l2Cache {
				if v.timestamp < oldestTime {
					oldestTime = v.timestamp
					oldestKey = k
				}
			}
			delete(c.l2Cache, oldestKey)
		}
		c.l2Mu.Unlock()

		// 存入 L3 缓存
		// 这里需要实现将缓存写入磁盘的逻辑
		// ...
	}()
}

// TwoStageSearchConfig 两阶段搜索配置
type TwoStageSearchConfig struct {
	CoarseK      int     // 粗筛阶段返回的候选数量
	CoarseNprobe int     // 粗筛阶段的 nprobe
	FineK        int     // 精排阶段最终返回数量
	UseGPU       bool    // 是否在精排阶段使用 GPU
	Threshold    float64 // 相似度阈值
}

// EnableMultiLevelCache 启用多级缓存
func (db *VectorDB) EnableMultiLevelCache(l1Size, l2Size int, l3Path string) {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.multiCache = NewMultiLevelCache(l1Size, l2Size, l3Path)
	log.Info("多级缓存已启用：L1=%d, L2=%d, L3=%s", l1Size, l2Size, l3Path)
}

// CachedSearch 带缓存的搜索
func (db *VectorDB) CachedSearch(query []float64, k int) ([]entity.Result, error) {
	// 生成查询键
	queryKey := db.generateQueryKey(query, k)

	// 尝试从缓存获取
	if db.multiCache != nil {
		if cachedIDs, found := db.multiCache.Get(queryKey); found {
			// 缓存命中，转换为 Result 格式
			results := make([]entity.Result, len(cachedIDs))
			for i, id := range cachedIDs {
				// 重新计算相似度（或从缓存中获取）
				similarity, _ := db.CalculateCosineSimilarity(id, query)
				results[i] = entity.Result{
					Id:         id,
					Similarity: similarity,
				}
			}
			return results, nil
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
	if db.multiCache != nil {
		ids := make([]string, len(results))
		for i, result := range results {
			ids[i] = result.Id
		}
		db.multiCache.Put(queryKey, ids)
	}

	return results, nil
}

// generateQueryKey 生成查询键
func (db *VectorDB) generateQueryKey(query []float64, k int) string {
	h := fnv.New64a()
	for _, val := range query {
		binary.Write(h, binary.LittleEndian, val)
	}
	binary.Write(h, binary.LittleEndian, int64(k))
	return fmt.Sprintf("%x", h.Sum64())
}
