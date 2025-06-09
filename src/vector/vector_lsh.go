package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"math"
	"sort"
	"sync"
	"time"
)

// LSH 配置结构
type LSHConfig struct {
	NumTables         int           `json:"num_tables"`         // 哈希表数量
	NumHashFunctions  int           `json:"num_hash_functions"` // 每表哈希函数数量
	HashFamilyType    LSHFamilyType `json:"hash_family_type"`   // 哈希族类型
	BucketSize        int           `json:"bucket_size"`        // 桶大小
	W                 float64       `json:"w"`                  // LSH 参数 w
	R                 float64       `json:"r"`                  // 查询半径
	AdaptiveThreshold float64       `json:"adaptive_threshold"` // 自适应阈值
	EnableMultiProbe  bool          `json:"enable_multi_probe"` // 启用多探测
	ProbeRadius       int           `json:"probe_radius"`       // 探测半径
}

// LSH 族类型
type LSHFamilyType int

const (
	LSHFamilyRandomProjection LSHFamilyType = iota // 随机投影
	LSHFamilyMinHash                               // MinHash
	LSHFamilyP2LSH                                 // p-stable LSH
	LSHFamilyAngular                               // 角度 LSH
	LSHFamilyEuclidean                             // 欧几里得 LSH
)

// EnhancedLSHIndex 增强 LSH 索引结构
type EnhancedLSHIndex struct {
	Tables        []EnhancedLSHTable `json:"tables"`
	Config        *LSHConfig         `json:"config"`
	Statistics    *LSHStatistics     `json:"statistics"`
	LastOptimized time.Time          `json:"last_optimized"`
	mu            sync.RWMutex
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
	Hash(vector []float64) uint64
	GetType() LSHFamilyType
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
			HashFamilyType:    LSHFamilyRandomProjection,
			BucketSize:        100,
			W:                 4.0,
			R:                 1.0,
			AdaptiveThreshold: 0.8,
			EnableMultiProbe:  true,
			ProbeRadius:       2,
		}
	}

	db.LshConfig = config

	// 1. 创建 LSH 表
	tables := make([]EnhancedLSHTable, config.NumTables)

	for i := 0; i < config.NumTables; i++ {
		// 2. 生成哈希函数
		hashFunctions := db.generateLSHHashFunctions(config)

		tables[i] = EnhancedLSHTable{
			ID:             i,
			HashFunctions:  hashFunctions,
			Buckets:        make(map[uint64]*LSHBucket),
			BucketStats:    make(map[uint64]*BucketStats),
			CollisionCount: 0,
			QueryCount:     0,
		}

		// 3. 插入所有向量
		for id, vector := range db.vectors {
			hashValue := db.computeEnhancedLSHHash(vector, hashFunctions)

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

	// 4. 创建增强索引
	db.LshIndex = &EnhancedLSHIndex{
		Tables:        tables,
		Config:        config,
		Statistics:    &LSHStatistics{},
		LastOptimized: time.Now(),
	}

	// 5. 初始化自适应 LSH
	db.AdaptiveLSH = &AdaptiveLSH{
		PerformanceHistory: make([]LSHPerformance, 0),
		OptimalParams:      config,
		LastTuning:         time.Now(),
		TuningInterval:     time.Hour,
	}

	log.Info("增强 LSH 索引构建完成，共 %d 个哈希表", config.NumTables)
	return nil
}

// EnhancedLSHSearch 增强 LSH 搜索
func (db *VectorDB) EnhancedLSHSearch(query []float64, k int) ([]entity.Result, error) {
	if db.LshIndex == nil {
		return db.lshSearch(query, k, 5)
	}

	db.LshIndex.mu.RLock()
	defer db.LshIndex.mu.RUnlock()

	startTime := time.Now()
	candidateSet := make(map[string]struct{})

	// 1. 多表查询
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

	// 5. 精确排序
	results := make([]entity.Result, 0, len(candidateSet))
	for id := range candidateSet {
		if vector, exists := db.vectors[id]; exists {
			similarity := acceler.CosineSimilarity(query, vector)
			results = append(results, entity.Result{
				Id:         id,
				Similarity: similarity,
			})
		}
	}

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

// LSH 族结构
type LSHFamily struct {
	ID               string                 `json:"id"`                 // LSH族唯一标识
	FamilyType       LSHFamilyType          `json:"family_type"`        // LSH族类型
	HashFunctions    []LSHHashFunction      `json:"hash_functions"`     // 哈希函数集合
	Parameters       map[string]interface{} `json:"parameters"`         // LSH族参数
	Dimension        int                    `json:"dimension"`          // 向量维度
	NumHashFunctions int                    `json:"num_hash_functions"` // 哈希函数数量
	W                float64                `json:"w"`                  // LSH参数w（用于p-stable LSH）
	R                float64                `json:"r"`                  // 查询半径
	CreatedAt        time.Time              `json:"created_at"`         // 创建时间
	LastUsed         time.Time              `json:"last_used"`          // 最后使用时间
	UsageCount       int64                  `json:"usage_count"`        // 使用次数
	Performance      *LSHFamilyPerformance  `json:"performance"`        // 性能指标
	mu               sync.RWMutex           // 并发控制锁
}

// LSH族性能指标
type LSHFamilyPerformance struct {
	AvgCollisionRate float64       `json:"avg_collision_rate"` // 平均碰撞率
	AvgRecall        float64       `json:"avg_recall"`         // 平均召回率
	AvgPrecision     float64       `json:"avg_precision"`      // 平均精确率
	AvgQueryTime     time.Duration `json:"avg_query_time"`     // 平均查询时间
	TotalQueries     int64         `json:"total_queries"`      // 总查询次数
	LastUpdated      time.Time     `json:"last_updated"`       // 最后更新时间
}

// NewLSHFamily 创建新的LSH族
func NewLSHFamily(id string, familyType LSHFamilyType, dimension int, numHashFunctions int, parameters map[string]interface{}) *LSHFamily {
	return &LSHFamily{
		ID:               id,
		FamilyType:       familyType,
		HashFunctions:    make([]LSHHashFunction, 0, numHashFunctions),
		Parameters:       parameters,
		Dimension:        dimension,
		NumHashFunctions: numHashFunctions,
		W:                getFloatParameter(parameters, "w", 4.0),
		R:                getFloatParameter(parameters, "r", 1.0),
		CreatedAt:        time.Now(),
		LastUsed:         time.Now(),
		UsageCount:       0,
		Performance: &LSHFamilyPerformance{
			AvgCollisionRate: 0.0,
			AvgRecall:        0.0,
			AvgPrecision:     0.0,
			AvgQueryTime:     0,
			TotalQueries:     0,
			LastUpdated:      time.Now(),
		},
	}
}

// UpdatePerformance 更新LSH族性能指标
func (lf *LSHFamily) UpdatePerformance(recall, precision float64, queryTime time.Duration, collisionRate float64) {
	lf.mu.Lock()
	defer lf.mu.Unlock()

	lf.LastUsed = time.Now()
	lf.UsageCount++

	if lf.Performance == nil {
		lf.Performance = &LSHFamilyPerformance{}
	}

	// 使用指数移动平均更新性能指标
	alpha := 0.1
	lf.Performance.AvgRecall = lf.Performance.AvgRecall*(1-alpha) + recall*alpha
	lf.Performance.AvgPrecision = lf.Performance.AvgPrecision*(1-alpha) + precision*alpha
	lf.Performance.AvgCollisionRate = lf.Performance.AvgCollisionRate*(1-alpha) + collisionRate*alpha

	if lf.Performance.AvgQueryTime == 0 {
		lf.Performance.AvgQueryTime = queryTime
	} else {
		lf.Performance.AvgQueryTime = time.Duration(float64(lf.Performance.AvgQueryTime)*(1-alpha) + float64(queryTime)*alpha)
	}

	lf.Performance.TotalQueries++
	lf.Performance.LastUpdated = time.Now()
}

// GetEffectiveness 获取LSH族的有效性评分
func (lf *LSHFamily) GetEffectiveness() float64 {
	lf.mu.RLock()
	defer lf.mu.RUnlock()

	if lf.Performance == nil || lf.Performance.TotalQueries == 0 {
		return 0.0
	}

	// 综合考虑召回率、精确率和查询时间
	recallWeight := 0.4
	precisionWeight := 0.4
	timeWeight := 0.2

	// 时间评分：查询时间越短评分越高（归一化到0-1）
	maxAcceptableTime := 100 * time.Millisecond
	timeScore := 1.0 - math.Min(1.0, float64(lf.Performance.AvgQueryTime)/float64(maxAcceptableTime))

	return lf.Performance.AvgRecall*recallWeight + lf.Performance.AvgPrecision*precisionWeight + timeScore*timeWeight
}

// IsExpired 检查LSH族是否过期（长时间未使用）
func (lf *LSHFamily) IsExpired(expireDuration time.Duration) bool {
	lf.mu.RLock()
	defer lf.mu.RUnlock()
	return time.Since(lf.LastUsed) > expireDuration
}

// 辅助函数：从参数映射中获取浮点数参数
func getFloatParameter(params map[string]interface{}, key string, defaultValue float64) float64 {
	if params == nil {
		return defaultValue
	}
	if val, exists := params[key]; exists {
		if floatVal, ok := val.(float64); ok {
			return floatVal
		}
	}
	return defaultValue
}
