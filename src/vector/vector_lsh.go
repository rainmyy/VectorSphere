package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/enum"
	hash2 "VectorSphere/src/library/hash"
	"VectorSphere/src/library/logger"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// LSHTable 局部敏感哈希表结构
type LSHTable struct {
	HashFunctions [][]float64         // 哈希函数参数
	Buckets       map[uint64][]string // 哈希桶，存储向量ID
}

// LSHConfig LSH 配置结构
type LSHConfig struct {
	NumTables         int                `json:"num_tables"`         // 哈希表数量
	NumHashFunctions  int                `json:"num_hash_functions"` // 每表哈希函数数量
	HashFamilyType    enum.LSHFamilyType `json:"hash_family_type"`   // 哈希族类型
	BucketSize        int                `json:"bucket_size"`        // 桶大小
	W                 float64            `json:"w"`                  // LSH 参数 w
	R                 float64            `json:"r"`                  // 查询半径
	AdaptiveThreshold float64            `json:"adaptive_threshold"` // 自适应阈值
	EnableMultiProbe  bool               `json:"enable_multi_probe"` // 启用多探测
	ProbeRadius       int                `json:"probe_radius"`       // 探测半径
}

// EnhancedLSHIndex 增强 LSH 索引结构
type EnhancedLSHIndex struct {
	Tables        []EnhancedLSHTable `json:"tables"`
	Config        *LSHConfig         `json:"config"`
	Statistics    *LSHStatistics     `json:"statistics"`
	LastOptimized time.Time          `json:"last_optimized"`
	mu            sync.RWMutex
	Enable        bool `json:"enable"`
}

// EnhancedLSHTable 增强 LSH 表结构
type EnhancedLSHTable struct {
	ID             int                     `json:"id"`
	HashFunctions  []LSHHashFunction       `json:"hash_functions"`
	Buckets        map[uint64]*LSHBucket   `json:"buckets"`
	BucketStats    map[uint64]*BucketStats `json:"bucket_stats"`
	CollisionCount int64                   `json:"collision_count"`
	QueryCount     int64                   `json:"query_count"`
	mu             sync.RWMutex
}

// LSHHashFunction LSH 哈希函数接口
type LSHHashFunction interface {
	Hash(vector []float64) (uint64, error)
	GetType() enum.LSHFamilyType
	GetParameters() map[string]interface{}
}

// LSHBucket LSH 桶结构
type LSHBucket struct {
	VectorIDs    []string  `json:"vector_ids"`
	LastAccessed time.Time `json:"last_accessed"`
	AccessCount  int64     `json:"access_count"`
	mu           sync.RWMutex
}

// BucketStats 桶统计信息
type BucketStats struct {
	Size         int           `json:"size"`
	AvgQueryTime time.Duration `json:"avg_query_time"`
	HitRate      float64       `json:"hit_rate"`
	LastUpdated  time.Time     `json:"last_updated"`
}

// LSHStatistics LSH 统计信息
type LSHStatistics struct {
	TotalQueries       int64       `json:"total_queries"`
	AvgCandidates      float64     `json:"avg_candidates"`
	AvgRecall          float64     `json:"avg_recall"`
	AvgPrecision       float64     `json:"avg_precision"`
	BucketDistribution map[int]int `json:"bucket_distribution"`
	CollisionRate      float64     `json:"collision_rate"`
}

// AdaptiveLSH 自适应 LSH
type AdaptiveLSH struct {
	PerformanceHistory []LSHPerformance `json:"performance_history"`
	OptimalParams      *LSHConfig       `json:"optimal_params"`
	LastTuning         time.Time        `json:"last_tuning"`
	TuningInterval     time.Duration    `json:"tuning_interval"`
}

// LSHPerformance LSH 性能记录
type LSHPerformance struct {
	Timestamp      time.Time     `json:"timestamp"`
	Config         *LSHConfig    `json:"config"`
	Recall         float64       `json:"recall"`
	Precision      float64       `json:"precision"`
	QueryTime      time.Duration `json:"query_time"`
	CandidateCount int           `json:"candidate_count"`
}

// BuildEnhancedLSHIndex 构建增强 LSH 索引
func (db *VectorDB) BuildEnhancedLSHIndex(config *LSHConfig) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if config == nil {
		config = &LSHConfig{
			NumTables:         10,
			NumHashFunctions:  8,
			HashFamilyType:    enum.LSHFamilyRandomProjection,
			BucketSize:        100,
			W:                 4.0,
			R:                 1.0,
			AdaptiveThreshold: 0.8,
			EnableMultiProbe:  true,
			ProbeRadius:       2,
		}
	}

	db.LshConfig = config

	// 获取向量维度
	vectorDimension := db.getVectorDimensionUnsafe()
	if vectorDimension == 0 {
		logger.Warning("Cannot determine vector dimension, using default dimension 128")
		vectorDimension = 128
	}

	// 1. 创建LSH族参数映射
	lshFamilyParams := map[string]interface{}{
		"w":                  config.W,
		"r":                  config.R,
		"bucket_size":        config.BucketSize,
		"adaptive_threshold": config.AdaptiveThreshold,
		"enable_multi_probe": config.EnableMultiProbe,
		"probe_radius":       config.ProbeRadius,
	}

	// 2. 创建LSH族实例
	lshFamily := NewLSHFamily(
		fmt.Sprintf("lsh_family_%d_%d", config.HashFamilyType, time.Now().Unix()),
		config.HashFamilyType,
		vectorDimension,
		config.NumHashFunctions,
		lshFamilyParams,
	)

	// 3. 生成哈希函数并添加到LSH族
	hashFunctions := db.generateLSHHashFunctionsWithFamily(config, lshFamily)
	lshFamily.HashFunctions = hashFunctions

	// 4. 创建 LSH 表
	tables := make([]EnhancedLSHTable, config.NumTables)

	for i := 0; i < config.NumTables; i++ {
		// 使用LSH族的哈希函数
		tables[i] = EnhancedLSHTable{
			ID:             i,
			HashFunctions:  lshFamily.HashFunctions,
			Buckets:        make(map[uint64]*LSHBucket),
			BucketStats:    make(map[uint64]*BucketStats),
			CollisionCount: 0,
			QueryCount:     0,
		}

		// 5. 插入所有向量
		for id, vector := range db.vectors {
			hashValue := db.computeEnhancedLSHHash(vector, lshFamily.HashFunctions)

			if tables[i].Buckets[hashValue] == nil {
				tables[i].Buckets[hashValue] = &LSHBucket{
					VectorIDs:    make([]string, 0),
					LastAccessed: time.Now(),
					AccessCount:  0,
				}
			}

			tables[i].Buckets[hashValue].VectorIDs = append(
				tables[i].Buckets[hashValue].VectorIDs, id)
		}
	}

	// 6. 创建增强索引
	db.LshIndex = &EnhancedLSHIndex{
		Tables:        tables,
		Config:        config,
		Statistics:    &LSHStatistics{BucketDistribution: make(map[int]int)},
		LastOptimized: time.Now(),
		Enable:        true,
	}

	// 7. 初始化自适应 LSH
	db.AdaptiveLSH = &AdaptiveLSH{
		PerformanceHistory: make([]LSHPerformance, 0),
		OptimalParams:      config,
		LastTuning:         time.Now(),
		TuningInterval:     time.Hour,
	}

	// 8. 存储LSH族实例以供后续使用
	if db.LshFamilies == nil {
		db.LshFamilies = make(map[string]*LSHFamily)
	}
	db.LshFamilies[lshFamily.ID] = lshFamily

	logger.Info("增强 LSH 索引构建完成，共 %d 个哈希表，使用LSH族：%s", config.NumTables, lshFamily.ID)
	return nil
}

// generateLSHHashFunctionsWithFamily 使用LSH族生成哈希函数
func (db *VectorDB) generateLSHHashFunctionsWithFamily(config *LSHConfig, lshFamily *LSHFamily) []LSHHashFunction {
	if config == nil || lshFamily == nil {
		logger.Warning("LSH config or family is nil, using fallback generation")
		return db.generateLSHHashFunctions(config)
	}

	vectorDimension := lshFamily.Dimension
	hashFunctions := make([]LSHHashFunction, 0, config.NumHashFunctions)

	// 根据哈希族类型生成相应的哈希函数
	switch config.HashFamilyType {
	case enum.LSHFamilyRandomProjection:
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机投影向量（高斯分布）
			projectionVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				projectionVector[j] = rand.NormFloat64()
			}

			// 生成随机偏移 b，均匀分布在 [0, w)
			b := rand.Float64() * lshFamily.W

			hashFunctions = append(hashFunctions, &hash2.RandomProjectionHash{
				ProjectionVector: projectionVector,
				W:                lshFamily.W,
				B:                b,
				FamilyType:       enum.LSHFamilyRandomProjection,
			})
		}
	case enum.LSHFamilyMinHash:
		// MinHash 实现
		for i := 0; i < config.NumHashFunctions; i++ {
			// 每个哈希函数使用不同数量的内部哈希函数
			numInternalHashes := 64 // 可以根据需要调整
			minHash := hash2.NewMinHash(numInternalHashes)
			hashFunctions = append(hashFunctions, minHash)
		}

	case enum.LSHFamilyAngular:
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机向量（高斯分布）
			randomVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				randomVector[j] = rand.NormFloat64()
			}

			hashFunctions = append(hashFunctions, &hash2.AngularHash{
				RandomVector: randomVector,
				FamilyType:   enum.LSHFamilyAngular,
			})
		}

	case enum.LSHFamilyEuclidean:
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机向量（高斯分布）
			randomVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				randomVector[j] = rand.NormFloat64()
			}

			// 生成随机偏移 b，均匀分布在 [0, w)
			b := rand.Float64() * lshFamily.W

			hashFunctions = append(hashFunctions, &hash2.EuclideanHash{
				RandomVector: randomVector,
				W:            lshFamily.W,
				B:            b,
				FamilyType:   enum.LSHFamilyEuclidean,
			})
		}

	case enum.LSHFamilyP2LSH:
		// p-stable LSH（类似欧几里得LSH）
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机向量（高斯分布）
			randomVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				randomVector[j] = rand.NormFloat64()
			}

			// 生成随机偏移 b，均匀分布在 [0, w)
			b := rand.Float64() * lshFamily.W

			hashFunctions = append(hashFunctions, &hash2.EuclideanHash{
				RandomVector: randomVector,
				W:            lshFamily.W,
				B:            b,
				FamilyType:   enum.LSHFamilyP2LSH,
			})
		}

	default:
		logger.Warning("Unsupported LSH family type: %d, using RandomProjection", config.HashFamilyType)
		// 默认使用随机投影
		for i := 0; i < config.NumHashFunctions; i++ {
			projectionVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				projectionVector[j] = rand.NormFloat64()
			}

			b := rand.Float64() * lshFamily.W

			hashFunctions = append(hashFunctions, &hash2.RandomProjectionHash{
				ProjectionVector: projectionVector,
				W:                lshFamily.W,
				B:                b,
				FamilyType:       enum.LSHFamilyRandomProjection,
			})
		}
	}

	// 更新LSH族的使用统计
	lshFamily.mu.Lock()
	lshFamily.LastUsed = time.Now()
	lshFamily.UsageCount++
	lshFamily.mu.Unlock()

	logger.Info("Generated %d LSH hash functions of type %d for dimension %d using family %s",
		len(hashFunctions), config.HashFamilyType, vectorDimension, lshFamily.ID)

	return hashFunctions
}

// computeEnhancedLSHHash 计算增强LSH哈希值
func (db *VectorDB) computeEnhancedLSHHash(vector []float64, hashFunctions []LSHHashFunction) uint64 {
	if len(hashFunctions) == 0 {
		logger.Warning("No hash functions provided for LSH hash computation")
		return 0
	}

	if len(vector) == 0 {
		logger.Warning("Empty vector provided for LSH hash computation")
		return 0
	}

	// 组合多个哈希函数的结果
	var combinedHash uint64 = 0
	for i, hashFunc := range hashFunctions {
		if hashFunc == nil {
			logger.Warning("Null hash function at index %d, skipping", i)
			continue
		}

		// 计算单个哈希函数的哈希值
		hashValue, err := hashFunc.Hash(vector)
		if err != nil {
			return combinedHash
		}

		// 使用位移和异或操作组合哈希值
		// 每个哈希函数的结果左移不同的位数，避免冲突
		shiftBits := uint(i % 64) // 防止位移超过64位
		combinedHash ^= (hashValue << shiftBits) | (hashValue >> (64 - shiftBits))
	}

	// 如果所有哈希函数都无效，返回向量的简单哈希
	if combinedHash == 0 {
		logger.Warning("All hash functions failed, using fallback hash")
		combinedHash = db.computeFallbackHash(vector)
	}

	return combinedHash
}

// computeFallbackHash 计算备用哈希值（当所有LSH哈希函数都失败时使用）
func (db *VectorDB) computeFallbackHash(vector []float64) uint64 {
	if len(vector) == 0 {
		return 0
	}

	// 使用简单的多项式滚动哈希
	var hash uint64 = 5381
	for i, val := range vector {
		// 将浮点数转换为整数进行哈希
		intVal := uint64(val * 1000000) // 保留6位小数精度
		hash = ((hash << 5) + hash) + intVal + uint64(i)
	}

	return hash
}

// generateLSHHashFunctions 生成LSH哈希函数
func (db *VectorDB) generateLSHHashFunctions(config *LSHConfig) []LSHHashFunction {
	if config == nil {
		logger.Warning("LSH config is nil, using default configuration")
		config = &LSHConfig{
			NumHashFunctions: 8,
			HashFamilyType:   enum.LSHFamilyRandomProjection,
			W:                4.0,
		}
	}

	// 获取向量维度
	vectorDimension := db.getVectorDimension()
	if vectorDimension == 0 {
		logger.Warning("Cannot determine vector dimension, using default dimension 128")
		vectorDimension = 128
	}

	hashFunctions := make([]LSHHashFunction, 0, config.NumHashFunctions)

	// 根据哈希族类型生成相应的哈希函数
	switch config.HashFamilyType {
	case enum.LSHFamilyRandomProjection:
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机投影向量（高斯分布）
			projectionVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				projectionVector[j] = rand.NormFloat64()
			}

			// 生成随机偏移 b，均匀分布在 [0, w)
			b := rand.Float64() * config.W

			hashFunctions = append(hashFunctions, &hash2.RandomProjectionHash{
				ProjectionVector: projectionVector,
				W:                config.W,
				B:                b,
				FamilyType:       enum.LSHFamilyRandomProjection,
			})
		}

	case enum.LSHFamilyMinHash:
		// MinHash 实现
		for i := 0; i < config.NumHashFunctions; i++ {
			numInternalHashes := 64 // 可以根据需要调整
			minHash := hash2.NewMinHash(numInternalHashes)
			hashFunctions = append(hashFunctions, minHash)
		}

	case enum.LSHFamilyAngular:
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机向量（高斯分布）
			randomVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				randomVector[j] = rand.NormFloat64()
			}

			hashFunctions = append(hashFunctions, &hash2.AngularHash{
				RandomVector: randomVector,
				FamilyType:   enum.LSHFamilyAngular,
			})
		}

	case enum.LSHFamilyEuclidean:
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机向量（高斯分布）
			randomVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				randomVector[j] = rand.NormFloat64()
			}

			// 生成随机偏移 b，均匀分布在 [0, w)
			b := rand.Float64() * config.W

			hashFunctions = append(hashFunctions, &hash2.EuclideanHash{
				RandomVector: randomVector,
				W:            config.W,
				B:            b,
				FamilyType:   enum.LSHFamilyEuclidean,
			})
		}

	case enum.LSHFamilyP2LSH:
		// p-stable LSH（类似欧几里得LSH）
		for i := 0; i < config.NumHashFunctions; i++ {
			// 生成随机向量（高斯分布）
			randomVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				randomVector[j] = rand.NormFloat64()
			}

			// 生成随机偏移 b，均匀分布在 [0, w)
			b := rand.Float64() * config.W

			hashFunctions = append(hashFunctions, &hash2.EuclideanHash{
				RandomVector: randomVector,
				W:            config.W,
				B:            b,
				FamilyType:   enum.LSHFamilyP2LSH,
			})
		}

	default:
		logger.Warning("Unsupported LSH family type: %d, using RandomProjection", config.HashFamilyType)
		// 默认使用随机投影
		for i := 0; i < config.NumHashFunctions; i++ {
			projectionVector := make([]float64, vectorDimension)
			for j := 0; j < vectorDimension; j++ {
				projectionVector[j] = rand.NormFloat64()
			}

			b := rand.Float64() * config.W

			hashFunctions = append(hashFunctions, &hash2.RandomProjectionHash{
				ProjectionVector: projectionVector,
				W:                config.W,
				B:                b,
				FamilyType:       enum.LSHFamilyRandomProjection,
			})
		}
	}

	logger.Info("Generated %d LSH hash functions of type %d for dimension %d",
		len(hashFunctions), config.HashFamilyType, vectorDimension)

	return hashFunctions
}

// getVectorDimension 获取向量维度
func (db *VectorDB) getVectorDimension() int {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return 0
	}

	// 从第一个向量获取维度
	for _, vector := range db.vectors {
		return len(vector)
	}

	return 0
}

// getVectorDimensionUnsafe 获取向量维度（不加锁版本，用于已经持有锁的情况）
func (db *VectorDB) getVectorDimensionUnsafe() int {
	if len(db.vectors) == 0 {
		return 0
	}

	// 从第一个向量获取维度
	for _, vector := range db.vectors {
		return len(vector)
	}

	return 0
}

// EnhancedLSHSearch 增强 LSH 搜索
func (db *VectorDB) EnhancedLSHSearch(query []float64, k int) ([]entity.Result, error) {
	// 检查查询向量是否为空或nil
	if query == nil || len(query) == 0 {
		return []entity.Result{}, nil
	}

	if db.LshIndex == nil {
		return db.lshSearch(query, k, 5)
	}

	startTime := time.Now()
	candidateSet := make(map[string]struct{})

	// 1. 多表查询
	db.LshIndex.mu.RLock()
	for i := range db.LshIndex.Tables {
		table := &db.LshIndex.Tables[i]

		// 2. 计算主哈希值
		hashValue := db.computeEnhancedLSHHash(query, table.HashFunctions)

		// 3. 添加主桶候选
		if bucket, exists := table.Buckets[hashValue]; exists {
			for _, id := range bucket.VectorIDs {
				candidateSet[id] = struct{}{}
			}

			// 更新访问统计
			bucket.mu.Lock()
			bucket.LastAccessed = time.Now()
			bucket.AccessCount++
			bucket.mu.Unlock()
		}

		// 4. 多探测搜索（如果启用）
		if db.LshConfig.EnableMultiProbe {
			probeHashes := db.generateProbeHashes(hashValue, db.LshConfig.ProbeRadius)
			for _, probeHash := range probeHashes {
				if bucket, exists := table.Buckets[probeHash]; exists {
					for _, id := range bucket.VectorIDs {
						candidateSet[id] = struct{}{}
					}
				}
			}
		}

		table.mu.Lock()
		table.QueryCount++
		table.mu.Unlock()
	}
	db.LshIndex.mu.RUnlock()

	// 5. 精确排序
	results := make([]entity.Result, 0, len(candidateSet))

	// 获取向量数据的读锁
	db.mu.RLock()
	for id := range candidateSet {
		if vector, exists := db.vectors[id]; exists {
			similarity := acceler.CosineSimilarity(query, vector)
			results = append(results, entity.Result{
				Id:         id,
				Similarity: similarity,
			})
		}
	}
	db.mu.RUnlock()

	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if k > len(results) {
		k = len(results)
	}

	// 6. 更新统计信息
	queryTime := time.Since(startTime)
	db.updateLSHStatistics(len(candidateSet), len(results), queryTime)

	return results[:k], nil
}

/*
- 更新总查询次数
- 使用指数移动平均更新平均候选数
- 更新桶分布统计
- 计算并更新碰撞率
- 记录性能数据到自适应LSH历史记录
- 检查是否需要进行自适应调整
*/
// updateLSHStatistics 更新LSH统计信息
func (db *VectorDB) updateLSHStatistics(candidatesCount, resultsCount int, queryTime time.Duration) {
	if db.LshIndex == nil || db.LshIndex.Statistics == nil {
		return
	}

	// 使用写锁保护统计信息的更新
	db.LshIndex.mu.Lock()
	defer db.LshIndex.mu.Unlock()

	stats := db.LshIndex.Statistics

	// 更新总查询次数
	stats.TotalQueries++

	// 使用指数移动平均更新平均候选数
	alpha := 0.1 // 平滑因子
	if stats.AvgCandidates == 0 {
		stats.AvgCandidates = float64(candidatesCount)
	} else {
		stats.AvgCandidates = stats.AvgCandidates*(1-alpha) + float64(candidatesCount)*alpha
	}

	// 更新桶分布统计
	if stats.BucketDistribution == nil {
		stats.BucketDistribution = make(map[int]int)
	}

	// 更新桶大小分布（按照10的倍数分组）
	bucketSizeGroup := (candidatesCount / 10) * 10
	stats.BucketDistribution[bucketSizeGroup]++

	// 计算碰撞率（候选集大小与结果集大小的比率）
	if candidatesCount > 0 && resultsCount > 0 {
		currentCollisionRate := float64(resultsCount) / float64(candidatesCount)
		if stats.CollisionRate == 0 {
			stats.CollisionRate = currentCollisionRate
		} else {
			stats.CollisionRate = stats.CollisionRate*(1-alpha) + currentCollisionRate*alpha
		}
	}

	// 记录性能数据到自适应LSH历史记录
	if db.AdaptiveLSH != nil {
		performanceRecord := LSHPerformance{
			Timestamp:      time.Now(),
			Config:         db.LshConfig,
			QueryTime:      queryTime,
			CandidateCount: candidatesCount,
			// 注意：实际召回率和精确率需要与真实结果比较才能计算
			// 这里简化处理，使用估计值
			Precision: stats.CollisionRate,
			Recall:    math.Min(1.0, float64(candidatesCount)/100.0), // 假设总共有100个相关项
		}

		// 限制历史记录大小
		maxHistorySize := 100
		if len(db.AdaptiveLSH.PerformanceHistory) >= maxHistorySize {
			// 移除最旧的记录
			db.AdaptiveLSH.PerformanceHistory = db.AdaptiveLSH.PerformanceHistory[1:]
		}

		db.AdaptiveLSH.PerformanceHistory = append(db.AdaptiveLSH.PerformanceHistory, performanceRecord)
	}

	// 记录日志
	logger.Trace("LSH查询统计：候选数=%d, 结果数=%d, 查询时间=%v, 平均候选数=%.2f, 碰撞率=%.2f",
		candidatesCount, resultsCount, queryTime, stats.AvgCandidates, stats.CollisionRate)

	// 检查是否需要自适应调整
	if db.AdaptiveLSH != nil && time.Since(db.AdaptiveLSH.LastTuning) > db.AdaptiveLSH.TuningInterval {
		go db.tuneAdaptiveLSH() // 异步执行调优，避免阻塞查询
	}
}

/*
- 分析性能历史记录，找出最佳参数组合
- 使用综合评分函数考虑召回率、精确率和查询时间
- 如果找到更好的配置，记录下来以供后续使用
*/
// tuneAdaptiveLSH 调整LSH参数以优化性能
func (db *VectorDB) tuneAdaptiveLSH() {
	if db.AdaptiveLSH == nil || len(db.AdaptiveLSH.PerformanceHistory) < 10 {
		return // 数据不足，无法进行有效调优
	}

	db.LshIndex.mu.Lock()
	defer db.LshIndex.mu.Unlock()

	// 分析性能历史记录，找出最佳参数组合
	bestScore := 0.0
	var bestConfig *LSHConfig

	// 简单评分函数：考虑召回率、精确率和查询时间的加权和
	for _, perf := range db.AdaptiveLSH.PerformanceHistory {
		// 时间评分：查询时间越短评分越高（归一化到0-1）
		maxAcceptableTime := 100 * time.Millisecond
		timeScore := 1.0 - math.Min(1.0, float64(perf.QueryTime)/float64(maxAcceptableTime))

		// 综合评分
		score := perf.Recall*0.4 + perf.Precision*0.4 + timeScore*0.2

		if score > bestScore {
			bestScore = score
			bestConfig = perf.Config
		}
	}

	// 如果找到更好的配置，记录下来（但不立即应用，避免频繁重建索引）
	if bestConfig != nil && bestScore > db.AdaptiveLSH.OptimalParams.AdaptiveThreshold {
		logger.Info("发现更优LSH配置，评分：%.2f，哈希表数：%d，哈希函数数：%d",
			bestScore, bestConfig.NumTables, bestConfig.NumHashFunctions)
		db.AdaptiveLSH.OptimalParams = bestConfig
	}

	db.AdaptiveLSH.LastTuning = time.Now()
}

/*
- 多种探测策略 ：

- 单位翻转：汉明距离为1的哈希值
- 双位翻转：汉明距离为2的哈希值（有限制地生成，避免组合爆炸）
- 位移探测：适用于基于投影的LSH，模拟连续空间中的邻近点
- 随机扰动：增加探测多样性
- 探测半径控制 ：根据传入的 probeRadius 参数控制探测范围和生成的哈希值数量
- 性能优化 ：

- 预估结果大小，避免频繁的切片扩容
- 对双位翻转进行限制，防止组合爆炸
- 结果去重，确保不重复探测同一个桶
*/
// generateProbeHashes 生成多探测哈希值
func (db *VectorDB) generateProbeHashes(baseHash uint64, probeRadius int) []uint64 {
	if probeRadius <= 0 {
		return []uint64{}
	}

	// 估计生成的哈希值数量，避免频繁的切片扩容
	// 每个位可以翻转，所以最多有 64 个邻居（位数）
	// 但实际上我们限制探测半径，所以数量会少一些
	estimatedSize := min(64, probeRadius*8) // 保守估计
	probeHashes := make([]uint64, 0, estimatedSize)

	// 1. 单位翻转（汉明距离为1的哈希值）
	if probeRadius >= 1 {
		for i := 0; i < 64; i++ {
			// 翻转第i位
			probeHash := baseHash ^ (1 << uint(i))
			probeHashes = append(probeHashes, probeHash)
		}
	}

	// 2. 双位翻转（汉明距离为2的哈希值）
	if probeRadius >= 2 {
		// 限制数量，避免组合爆炸
		maxBits := 8 // 最多考虑8位，否则组合太多
		for i := 0; i < maxBits; i++ {
			for j := i + 1; j < maxBits; j++ {
				// 同时翻转第i位和第j位
				probeHash := baseHash ^ ((1 << uint(i)) | (1 << uint(j)))
				probeHashes = append(probeHashes, probeHash)
			}
		}
	}

	// 3. 位移探测（适用于基于投影的LSH）
	// 对哈希值进行小幅度的位移，模拟连续空间中的邻近点
	probeHashes = append(probeHashes, baseHash+1, baseHash-1) // 加减1

	// 对于更大的探测半径，可以考虑更远的位移
	if probeRadius >= 3 {
		probeHashes = append(probeHashes, baseHash+2, baseHash-2) // 加减2
		probeHashes = append(probeHashes, baseHash+3, baseHash-3) // 加减3
	}

	// 4. 随机扰动（增加探测多样性）
	if probeRadius >= 4 {
		rand.New(rand.NewSource(time.Now().UnixNano()))
		for i := 0; i < 5; i++ { // 添加5个随机扰动的哈希值
			// 随机翻转1-3位
			mask := uint64(0)
			flipCount := rand.Intn(3) + 1
			for j := 0; j < flipCount; j++ {
				bitPos := rand.Intn(64)
				mask |= (1 << uint(bitPos))
			}
			probeHash := baseHash ^ mask
			probeHashes = append(probeHashes, probeHash)
		}
	}

	// 去重
	deduped := make(map[uint64]struct{})
	for _, hash := range probeHashes {
		deduped[hash] = struct{}{}
	}

	// 删除基础哈希值（如果存在）
	delete(deduped, baseHash)

	// 转回切片
	result := make([]uint64, 0, len(deduped))
	for hash := range deduped {
		result = append(result, hash)
	}

	logger.Trace("Generated %d probe hashes for base hash %d with radius %d",
		len(result), baseHash, probeRadius)

	return result
}
