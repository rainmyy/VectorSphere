package vector

import (
	"crypto/md5"
	"fmt"
	"math"
	"sort"
	"time"

	"VectorSphere/src/library/entity"
)

// generateCacheKey 生成缓存键
func (db *VectorDB) generateCacheKey(query []float64, k int, options SearchOptions) string {
	// 使用查询向量的哈希、k值和关键选项生成缓存键
	hash := md5.New()
	for _, v := range query {
		hash.Write([]byte(fmt.Sprintf("%.6f", v)))
	}
	hash.Write([]byte(fmt.Sprintf("%d", k)))
	hash.Write([]byte(fmt.Sprintf("%s", options.ForceStrategy)))
	hash.Write([]byte(fmt.Sprintf("%.3f", options.QualityLevel)))
	hash.Write([]byte(fmt.Sprintf("%d", options.Nprobe)))
	
	return fmt.Sprintf("%x", hash.Sum(nil))
}

// deduplicateResults 结果去重
func (db *VectorDB) deduplicateResults(results []entity.Result) []entity.Result {
	if len(results) <= 1 {
		return results
	}

	seen := make(map[string]bool)
	deduped := make([]entity.Result, 0, len(results))
	
	for _, result := range results {
		if !seen[result.Id] {
			seen[result.Id] = true
			deduped = append(deduped, result)
		}
	}
	
	return deduped
}

// reRankResults 使用精确距离计算重排结果
func (db *VectorDB) reRankResults(query []float64, candidates []entity.Result, k int) []entity.Result {
	if len(candidates) <= k {
		return candidates
	}

	// 重新计算精确距离
	for i := range candidates {
		if vector, exists := db.vectors[candidates[i].Id]; exists {
			candidates[i].Distance = db.calculateDistance(query, vector)
		}
	}

	// 按距离排序
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	// 返回前k个结果
	if len(candidates) > k {
		return candidates[:k]
	}
	return candidates
}

// refinePQResults 使用PQ解码进行精确重排
func (db *VectorDB) refinePQResults(query []float64, candidates []entity.Result, k int) []entity.Result {
	if db.pqCodebook == nil || len(candidates) <= k {
		return candidates
	}

	// 使用PQ解码重新计算距离
	for i := range candidates {
		if vector, exists := db.vectors[candidates[i].Id]; exists {
			// 使用原始向量计算精确距离（PQ相关功能暂时使用原始向量）
			candidates[i].Distance = db.calculateDistance(query, vector)
		}
	}

	// 按距离排序并返回前k个
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	if len(candidates) > k {
		return candidates[:k]
	}
	return candidates
}

// executeSearch 执行具体的搜索策略
func (db *VectorDB) executeSearch(query []float64, k int, strategy IndexStrategy, ctx SearchContext) ([]entity.Result, error) {
	switch strategy {
	case StrategyBruteForce:
		return db.bruteForceSearch(query, k)
	case StrategyIVF:
		return db.ivfSearchWithScores(query, k, ctx.Nprobe, db.GetOptimalStrategy(query))
	case StrategyHNSW:
		return db.hnswSearchWithScores(query, k)
	case StrategyPQ:
		return db.pqSearchWithScores(query, k)
	case StrategyHybrid:
		return db.hybridSearchWithScores(query, k, ctx)
	case StrategyEnhancedIVF:
		return db.EnhancedIVFSearch(query, k, ctx.Nprobe)
	case StrategyEnhancedLSH:
		return db.EnhancedLSHSearch(query, k)
	case StrategyIVFHNSW:
		return db.ivfHnswSearchWithScores(query, k, ctx)
	default:
		return db.ivfSearchWithScores(query, k, ctx.Nprobe, db.GetOptimalStrategy(query))
	}
}

// calculateDistance 计算两个向量之间的距离
func (db *VectorDB) calculateDistance(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return math.Inf(1)
	}

	var sum float64
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// decodePQVector 解码PQ向量（占位符实现）
func (db *VectorDB) decodePQVector(codes []byte) []float64 {
	// 这里应该实现实际的PQ解码逻辑
	// 目前返回空向量作为占位符
	return make([]float64, db.vectorDim)
}

// minInt 返回两个整数中的较小值
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// maxInt 返回两个整数中的较大值
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// AdaptiveSearchOptions 自适应搜索选项
type AdaptiveSearchOptions struct {
	// 自动调整参数
	AutoTuneNprobe   bool          // 自动调整nprobe
	AutoTuneEf       bool          // 自动调整ef参数
	AutoTuneTimeout  bool          // 自动调整超时时间
	TargetLatency    time.Duration // 目标延迟
	TargetRecall     float64       // 目标召回率
	
	// 负载均衡
	EnableLoadBalance bool // 启用负载均衡
	MaxConcurrency   int  // 最大并发数
	
	// 预热策略
	EnableWarmup     bool // 启用预热
	WarmupQueries    int  // 预热查询数量
}

// SearchMetrics 搜索指标
type SearchMetrics struct {
	Latency       time.Duration `json:"latency"`
	Throughput    float64       `json:"throughput"`
	Recall        float64       `json:"recall"`
	Precision     float64       `json:"precision"`
	CacheHitRate  float64       `json:"cache_hit_rate"`
	ErrorRate     float64       `json:"error_rate"`
	ResourceUsage ResourceUsage `json:"resource_usage"`
}

// ResourceUsage 资源使用情况
type ResourceUsage struct {
	CPUUsage    float64 `json:"cpu_usage"`
	MemoryUsage float64 `json:"memory_usage"`
	GPUUsage    float64 `json:"gpu_usage"`
	DiskIO      float64 `json:"disk_io"`
	NetworkIO   float64 `json:"network_io"`
}

// QualityMetrics 质量指标
type QualityMetrics struct {
	Recall    float64 `json:"recall"`
	Precision float64 `json:"precision"`
	F1Score   float64 `json:"f1_score"`
	NDCG      float64 `json:"ndcg"`
	MRR       float64 `json:"mrr"`
}

// CalculateQualityMetrics 计算质量指标
func (db *VectorDB) CalculateQualityMetrics(results []entity.Result, groundTruth []entity.Result) QualityMetrics {
	if len(results) == 0 || len(groundTruth) == 0 {
		return QualityMetrics{}
	}

	// 计算召回率
	recall := db.calculateRecall(results, groundTruth)
	
	// 计算精确率
	precision := db.calculatePrecision(results, groundTruth)
	
	// 计算F1分数
	f1Score := 0.0
	if recall+precision > 0 {
		f1Score = 2 * (recall * precision) / (recall + precision)
	}
	
	// 计算NDCG
	ndcg := db.calculateNDCG(results, groundTruth)
	
	// 计算MRR
	mrr := db.calculateMRR(results, groundTruth)

	return QualityMetrics{
		Recall:    recall,
		Precision: precision,
		F1Score:   f1Score,
		NDCG:      ndcg,
		MRR:       mrr,
	}
}

// calculateRecall 计算召回率
func (db *VectorDB) calculateRecall(results []entity.Result, groundTruth []entity.Result) float64 {
	if len(groundTruth) == 0 {
		return 0.0
	}

	truthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		truthSet[gt.Id] = true
	}

	hits := 0
	for _, result := range results {
		if truthSet[result.Id] {
			hits++
		}
	}

	return float64(hits) / float64(len(groundTruth))
}

// calculatePrecision 计算精确率
func (db *VectorDB) calculatePrecision(results []entity.Result, groundTruth []entity.Result) float64 {
	if len(results) == 0 {
		return 0.0
	}

	truthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		truthSet[gt.Id] = true
	}

	hits := 0
	for _, result := range results {
		if truthSet[result.Id] {
			hits++
		}
	}

	return float64(hits) / float64(len(results))
}

// calculateNDCG 计算归一化折扣累积增益
func (db *VectorDB) calculateNDCG(results []entity.Result, groundTruth []entity.Result) float64 {
	// 简化的NDCG计算
	if len(results) == 0 || len(groundTruth) == 0 {
		return 0.0
	}

	truthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		truthSet[gt.Id] = true
	}

	dcg := 0.0
	for i, result := range results {
		if truthSet[result.Id] {
			dcg += 1.0 / math.Log2(float64(i+2))
		}
	}

	// 理想DCG
	idcg := 0.0
	for i := 0; i < min(len(results), len(groundTruth)); i++ {
		idcg += 1.0 / math.Log2(float64(i+2))
	}

	if idcg == 0 {
		return 0.0
	}
	return dcg / idcg
}

// calculateMRR 计算平均倒数排名
func (db *VectorDB) calculateMRR(results []entity.Result, groundTruth []entity.Result) float64 {
	if len(results) == 0 || len(groundTruth) == 0 {
		return 0.0
	}

	truthSet := make(map[string]bool)
	for _, gt := range groundTruth {
		truthSet[gt.Id] = true
	}

	for i, result := range results {
		if truthSet[result.Id] {
			return 1.0 / float64(i+1)
		}
	}

	return 0.0
}