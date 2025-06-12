package vector

import (
	"VectorSphere/src/library/enum"
	"math"
	"sync"
	"time"
)

// LSHFamily LSH 族结构
type LSHFamily struct {
	ID               string                 `json:"id"`                 // LSH族唯一标识
	FamilyType       enum.LSHFamilyType     `json:"family_type"`        // LSH族类型
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

// LSHFamilyPerformance LSH族性能指标
type LSHFamilyPerformance struct {
	AvgCollisionRate float64       `json:"avg_collision_rate"` // 平均碰撞率
	AvgRecall        float64       `json:"avg_recall"`         // 平均召回率
	AvgPrecision     float64       `json:"avg_precision"`      // 平均精确率
	AvgQueryTime     time.Duration `json:"avg_query_time"`     // 平均查询时间
	TotalQueries     int64         `json:"total_queries"`      // 总查询次数
	LastUpdated      time.Time     `json:"last_updated"`       // 最后更新时间
}

// NewLSHFamily 创建新的LSH族
func NewLSHFamily(id string, familyType enum.LSHFamilyType, dimension int, numHashFunctions int, parameters map[string]interface{}) *LSHFamily {
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
