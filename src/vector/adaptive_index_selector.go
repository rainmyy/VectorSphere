package vector

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"math"
	"sync"
	"time"
)

// 添加自适应索引选择器
type AdaptiveIndexSelector struct {
	db                   *VectorDB
	performanceWindow    []PerformanceRecord
	windowSize           int
	lastOptimization     time.Time
	optimizationInterval time.Duration
	mu                   sync.RWMutex
}

type PerformanceRecord struct {
	Strategy    IndexStrategy
	Latency     time.Duration
	Quality     float64
	Timestamp   time.Time
	VectorCount int
	Dimension   int
}

// NewAdaptiveIndexSelector 创建自适应索引选择器
func NewAdaptiveIndexSelector(db *VectorDB) *AdaptiveIndexSelector {
	return &AdaptiveIndexSelector{
		db:                   db,
		performanceWindow:    make([]PerformanceRecord, 0),
		windowSize:           100, // 保留最近100次查询的性能记录
		lastOptimization:     time.Now(),
		optimizationInterval: 5 * time.Minute, // 每5分钟优化一次
	}
}

// RecordPerformance 记录性能数据
func (ais *AdaptiveIndexSelector) RecordPerformance(strategy IndexStrategy, latency time.Duration, quality float64, vectorCount, dimension int) {
	ais.mu.Lock()
	defer ais.mu.Unlock()

	record := PerformanceRecord{
		Strategy:    strategy,
		Latency:     latency,
		Quality:     quality,
		Timestamp:   time.Now(),
		VectorCount: vectorCount,
		Dimension:   dimension,
	}

	ais.performanceWindow = append(ais.performanceWindow, record)

	// 保持窗口大小
	if len(ais.performanceWindow) > ais.windowSize {
		ais.performanceWindow = ais.performanceWindow[1:]
	}

	// 检查是否需要优化
	if time.Since(ais.lastOptimization) > ais.optimizationInterval {
		go ais.optimizeStrategySelection()
	}
}

// optimizeStrategySelection 优化策略选择
func (ais *AdaptiveIndexSelector) optimizeStrategySelection() {
	ais.mu.Lock()
	defer ais.mu.Unlock()

	if len(ais.performanceWindow) < 10 {
		return // 数据不足，无法优化
	}

	// 分析不同策略的性能表现
	strategyPerf := make(map[IndexStrategy][]PerformanceRecord)
	for _, record := range ais.performanceWindow {
		strategyPerf[record.Strategy] = append(strategyPerf[record.Strategy], record)
	}

	// 计算每种策略的平均性能
	for strategy, records := range strategyPerf {
		if len(records) < 3 {
			continue // 样本太少
		}

		var totalLatency time.Duration
		var totalQuality float64
		for _, record := range records {
			totalLatency += record.Latency
			totalQuality += record.Quality
		}

		avgLatency := totalLatency / time.Duration(len(records))
		avgQuality := totalQuality / float64(len(records))

		log.Info("策略 %v 性能统计: 平均延迟=%v, 平均质量=%.3f, 样本数=%d",
			strategy, avgLatency, avgQuality, len(records))
	}

	ais.lastOptimization = time.Now()
}

// InitializeAdaptiveSelector 初始化自适应索引选择器
func (db *VectorDB) InitializeAdaptiveSelector() {
	if db.adaptiveSelector == nil {
		db.adaptiveSelector = NewAdaptiveIndexSelector(db)
		log.Info("自适应索引选择器已初始化")
	}
}

// GetPerformanceInsights 获取性能洞察
func (db *VectorDB) GetPerformanceInsights() map[string]interface{} {
	insights := make(map[string]interface{})

	// 基础统计信息
	db.statsMu.RLock()
	stats := db.stats
	db.statsMu.RUnlock()

	insights["basic_stats"] = map[string]interface{}{
		"total_queries":     stats.TotalQueries,
		"cache_hits":        stats.CacheHits,
		"avg_query_time":    stats.AvgQueryTime.String(),
		"memory_usage_mb":   float64(stats.MemoryUsage) / 1024 / 1024,
		"cache_hit_rate":    float64(stats.CacheHits) / float64(stats.TotalQueries),
		"last_reindex_time": stats.LastReindexTime.Format(time.RFC3339),
	}

	// 策略性能对比
	if db.strategySelector != nil && db.strategySelector.performance != nil {
		strategyPerf := make(map[string]interface{})
		for strategy, metrics := range db.strategySelector.performance {
			strategyName := db.getStrategyName(strategy)
			strategyPerf[strategyName] = map[string]interface{}{
				"avg_latency":    metrics.AvgLatency.String(),
				"throughput_qps": metrics.ThroughputQPS,
				"recall":         metrics.Recall,
				"memory_usage":   metrics.MemoryUsage,
				"last_updated":   metrics.LastUpdated.Format(time.RFC3339),
			}
		}
		insights["strategy_performance"] = strategyPerf
	}

	// 自适应选择器洞察
	if db.adaptiveSelector != nil {
		insights["adaptive_insights"] = db.adaptiveSelector.GetInsights()
	}

	return insights
}

// selectOptimalStrategyWithAdaptive 使用自适应选择器优化策略选择
func (db *VectorDB) selectOptimalStrategyWithAdaptive(ctx SearchContext) IndexStrategy {
	// 如果自适应选择器可用，优先使用其建议
	if db.adaptiveSelector != nil {
		if suggestion := db.adaptiveSelector.GetOptimalStrategy(ctx); suggestion != -1 {
			log.Trace("自适应选择器建议策略: %v", suggestion)
			return suggestion
		}
	}

	// 回退到原有的策略选择逻辑
	return db.SelectOptimalIndexStrategy(ctx)
}

// updateEnhancedPerformanceMetrics 增强的性能指标更新
func (db *VectorDB) updateEnhancedPerformanceMetrics(strategy IndexStrategy, latency time.Duration, resultCount int, quality float64, ctx SearchContext) {
	// 更新基础性能指标
	db.updatePerformanceMetrics(strategy, latency, resultCount)

	// 如果自适应选择器存在，记录详细性能数据
	if db.adaptiveSelector != nil {
		db.mu.RLock()
		vectorCount := len(db.vectors)
		db.mu.RUnlock()

		dimension := len(ctx.QueryVector)
		db.adaptiveSelector.RecordPerformance(strategy, latency, quality, vectorCount, dimension)
	}

	// 记录上下文相关的性能数据
	db.recordContextualPerformance(strategy, latency, quality, ctx)
}

// recordContextualPerformance 记录上下文相关的性能数据
func (db *VectorDB) recordContextualPerformance(strategy IndexStrategy, latency time.Duration, quality float64, ctx SearchContext) {
	db.statsMu.Lock()
	defer db.statsMu.Unlock()

	// 根据质量等级分类记录性能
	qualityLevel := "low"
	if ctx.QualityLevel > 0.8 {
		qualityLevel = "high"
	} else if ctx.QualityLevel > 0.6 {
		qualityLevel = "medium"
	}

	// 根据数据规模分类记录性能
	db.mu.RLock()
	vectorCount := len(db.vectors)
	db.mu.RUnlock()

	scaleLevel := "small"
	if vectorCount > 100000 {
		scaleLevel = "large"
	} else if vectorCount > 10000 {
		scaleLevel = "medium"
	}

	// 记录详细的性能日志
	log.Debug("上下文性能记录 - 策略: %v, 延迟: %v, 质量: %.3f, 质量等级: %s, 数据规模: %s(%d), 维度: %d",
		strategy, latency, quality, qualityLevel, scaleLevel, vectorCount, len(ctx.QueryVector))

	// 可以在这里添加更多的性能分析逻辑
	// 例如：异常检测、性能趋势分析等
	if latency > 5*time.Second {
		log.Warning("检测到异常长的查询延迟: %v, 策略: %v, 数据量: %d", latency, strategy, vectorCount)
	}

	if quality < 0.3 {
		log.Warning("检测到低质量搜索结果: %.3f, 策略: %v, 质量要求: %.2f", quality, strategy, ctx.QualityLevel)
	}
}

// estimateSearchQuality 估算搜索质量
func (db *VectorDB) estimateSearchQuality(results []entity.Result, ctx SearchContext) float64 {
	if len(results) == 0 {
		return 0.0
	}

	// 基于多个因素估算质量
	qualityScore := 0.0

	// 1. 结果数量质量 (期望返回k个结果)
	expectedCount := float64(ctx.K)
	actualCount := float64(len(results))
	countQuality := math.Min(1.0, actualCount/expectedCount)

	// 2. 相似度分布质量 (结果应该有合理的相似度分布)
	similarityQuality := 0.0
	if len(results) > 0 {
		// 计算相似度的方差，方差越小说明结果越集中，质量可能越高
		meanSim := 0.0
		for _, result := range results {
			meanSim += result.Similarity
		}
		meanSim /= float64(len(results))

		variance := 0.0
		for _, result := range results {
			diff := result.Similarity - meanSim
			variance += diff * diff
		}
		variance /= float64(len(results))

		// 将方差转换为质量分数 (方差越小，质量越高)
		similarityQuality = 1.0 / (1.0 + variance)

		// 如果最高相似度很低，降低质量分数
		if len(results) > 0 && results[0].Similarity < 0.5 {
			similarityQuality *= results[0].Similarity * 2 // 惩罚低相似度
		}
	}

	// 3. 响应时间质量 (基于超时设置)
	timeQuality := 1.0
	if ctx.Timeout > 0 {
		// 这里需要从外部传入实际耗时，暂时使用固定值
		// 在实际调用中应该传入latency参数
		timeQuality = 0.8 // 简化处理
	}

	// 综合质量分数 (加权平均)
	qualityScore = countQuality*0.4 + similarityQuality*0.5 + timeQuality*0.1

	return math.Min(1.0, math.Max(0.0, qualityScore))
}

// GetInsights 获取自适应选择器的洞察信息
func (ais *AdaptiveIndexSelector) GetInsights() map[string]interface{} {
	ais.mu.RLock()
	defer ais.mu.RUnlock()

	insights := make(map[string]interface{})

	// 基础统计信息
	insights["window_size"] = ais.windowSize
	insights["current_records"] = len(ais.performanceWindow)
	insights["last_optimization"] = ais.lastOptimization.Format(time.RFC3339)
	insights["optimization_interval"] = ais.optimizationInterval.String()

	if len(ais.performanceWindow) == 0 {
		insights["status"] = "no_data"
		return insights
	}

	// 策略使用统计
	strategyStats := make(map[string]interface{})
	strategyCount := make(map[IndexStrategy]int)
	strategyLatency := make(map[IndexStrategy][]time.Duration)
	strategyQuality := make(map[IndexStrategy][]float64)

	// 收集各策略的性能数据
	for _, record := range ais.performanceWindow {
		strategyCount[record.Strategy]++
		strategyLatency[record.Strategy] = append(strategyLatency[record.Strategy], record.Latency)
		strategyQuality[record.Strategy] = append(strategyQuality[record.Strategy], record.Quality)
	}

	// 计算各策略的详细统计
	for strategy, count := range strategyCount {
		strategyName := ais.getStrategyName(strategy)
		usagePercent := float64(count) / float64(len(ais.performanceWindow)) * 100

		// 计算平均延迟
		var totalLatency time.Duration
		for _, latency := range strategyLatency[strategy] {
			totalLatency += latency
		}
		avgLatency := totalLatency / time.Duration(count)

		// 计算平均质量
		var totalQuality float64
		for _, quality := range strategyQuality[strategy] {
			totalQuality += quality
		}
		avgQuality := totalQuality / float64(count)

		// 计算延迟标准差
		var latencyVariance float64
		avgLatencyFloat := float64(avgLatency)
		for _, latency := range strategyLatency[strategy] {
			diff := float64(latency) - avgLatencyFloat
			latencyVariance += diff * diff
		}
		latencyVariance /= float64(count)
		latencyStdDev := math.Sqrt(latencyVariance)

		// 计算质量标准差
		var qualityVariance float64
		for _, quality := range strategyQuality[strategy] {
			diff := quality - avgQuality
			qualityVariance += diff * diff
		}
		qualityVariance /= float64(count)
		qualityStdDev := math.Sqrt(qualityVariance)

		strategyStats[strategyName] = map[string]interface{}{
			"usage_count":      count,
			"usage_percentage": usagePercent,
			"avg_latency":      avgLatency.String(),
			"latency_std_dev":  time.Duration(latencyStdDev).String(),
			"avg_quality":      avgQuality,
			"quality_std_dev":  qualityStdDev,
			"min_latency":      ais.getMinLatency(strategyLatency[strategy]).String(),
			"max_latency":      ais.getMaxLatency(strategyLatency[strategy]).String(),
			"min_quality":      ais.getMinQuality(strategyQuality[strategy]),
			"max_quality":      ais.getMaxQuality(strategyQuality[strategy]),
		}
	}
	insights["strategy_statistics"] = strategyStats

	// 性能趋势分析
	trendAnalysis := ais.analyzeTrends()
	insights["trend_analysis"] = trendAnalysis

	// 推荐策略
	recommendation := ais.getStrategyRecommendation()
	insights["recommendation"] = recommendation

	// 数据质量评估
	dataQuality := ais.assessDataQuality()
	insights["data_quality"] = dataQuality

	// 异常检测
	anomalies := ais.detectAnomalies()
	insights["anomalies"] = anomalies

	return insights
}

// getStrategyName 获取策略名称
func (ais *AdaptiveIndexSelector) getStrategyName(strategy IndexStrategy) string {
	switch strategy {
	case StrategyBruteForce:
		return "BruteForce"
	case StrategyIVF:
		return "IVF"
	case StrategyHNSW:
		return "HNSW"
	case StrategyPQ:
		return "PQ"
	case StrategyHybrid:
		return "Hybrid"
	case EnhancedIVF:
		return "EnhancedIVF"
	case EnhancedLSH:
		return "EnhancedLSH"
	default:
		return "Unknown"
	}
}

// getMinLatency 获取最小延迟
func (ais *AdaptiveIndexSelector) getMinLatency(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	duration := latencies[0]
	for _, latency := range latencies[1:] {
		if latency < duration {
			duration = latency
		}
	}
	return duration
}

// getMaxLatency 获取最大延迟
func (ais *AdaptiveIndexSelector) getMaxLatency(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	duration := latencies[0]
	for _, latency := range latencies[1:] {
		if latency > duration {
			duration = latency
		}
	}
	return duration
}

// getMinQuality 获取最小质量
func (ais *AdaptiveIndexSelector) getMinQuality(qualities []float64) float64 {
	if len(qualities) == 0 {
		return 0
	}
	m := qualities[0]
	for _, quality := range qualities[1:] {
		if quality < m {
			m = quality
		}
	}
	return m
}

// getMaxQuality 获取最大质量
func (ais *AdaptiveIndexSelector) getMaxQuality(qualities []float64) float64 {
	if len(qualities) == 0 {
		return 0
	}
	m := qualities[0]
	for _, quality := range qualities[1:] {
		if quality > m {
			m = quality
		}
	}
	return m
}

// analyzeTrends 分析性能趋势
func (ais *AdaptiveIndexSelector) analyzeTrends() map[string]interface{} {
	if len(ais.performanceWindow) < 10 {
		return map[string]interface{}{
			"status":  "insufficient_data",
			"message": "需要至少10个数据点进行趋势分析",
		}
	}

	// 将数据分为前半部分和后半部分进行对比
	midPoint := len(ais.performanceWindow) / 2
	firstHalf := ais.performanceWindow[:midPoint]
	secondHalf := ais.performanceWindow[midPoint:]

	// 计算前半部分的平均性能
	var firstHalfLatency time.Duration
	var firstHalfQuality float64
	for _, record := range firstHalf {
		firstHalfLatency += record.Latency
		firstHalfQuality += record.Quality
	}
	firstHalfLatency /= time.Duration(len(firstHalf))
	firstHalfQuality /= float64(len(firstHalf))

	// 计算后半部分的平均性能
	var secondHalfLatency time.Duration
	var secondHalfQuality float64
	for _, record := range secondHalf {
		secondHalfLatency += record.Latency
		secondHalfQuality += record.Quality
	}
	secondHalfLatency /= time.Duration(len(secondHalf))
	secondHalfQuality /= float64(len(secondHalf))

	// 计算变化趋势
	latencyChange := float64(secondHalfLatency-firstHalfLatency) / float64(firstHalfLatency) * 100
	qualityChange := (secondHalfQuality - firstHalfQuality) / firstHalfQuality * 100

	trendStatus := "stable"
	if math.Abs(latencyChange) > 20 || math.Abs(qualityChange) > 10 {
		if latencyChange > 0 || qualityChange < 0 {
			trendStatus = "degrading"
		} else {
			trendStatus = "improving"
		}
	}

	return map[string]interface{}{
		"status":                  trendStatus,
		"latency_change_percent":  latencyChange,
		"quality_change_percent":  qualityChange,
		"first_half_avg_latency":  firstHalfLatency.String(),
		"second_half_avg_latency": secondHalfLatency.String(),
		"first_half_avg_quality":  firstHalfQuality,
		"second_half_avg_quality": secondHalfQuality,
		"analysis_period": map[string]interface{}{
			"start": firstHalf[0].Timestamp.Format(time.RFC3339),
			"end":   ais.performanceWindow[len(ais.performanceWindow)-1].Timestamp.Format(time.RFC3339),
		},
	}
}

// getStrategyRecommendation 获取策略推荐
func (ais *AdaptiveIndexSelector) getStrategyRecommendation() map[string]interface{} {
	if len(ais.performanceWindow) < 5 {
		return map[string]interface{}{
			"status":  "insufficient_data",
			"message": "需要更多数据进行策略推荐",
		}
	}

	// 计算各策略的综合评分
	strategyScores := make(map[IndexStrategy]float64)
	strategyCount := make(map[IndexStrategy]int)

	for _, record := range ais.performanceWindow {
		// 综合评分 = 质量权重 * 质量分数 + 速度权重 * 速度分数
		// 速度分数 = 1 / (1 + 标准化延迟)
		latencyScore := 1.0 / (1.0 + float64(record.Latency.Milliseconds())/1000.0)
		compositeScore := 0.6*record.Quality + 0.4*latencyScore

		strategyScores[record.Strategy] += compositeScore
		strategyCount[record.Strategy]++
	}

	// 计算平均评分
	var bestStrategy IndexStrategy
	var bestScore float64
	var strategyRankings []map[string]interface{}

	for strategy, totalScore := range strategyScores {
		avgScore := totalScore / float64(strategyCount[strategy])
		strategyName := ais.getStrategyName(strategy)

		strategyRankings = append(strategyRankings, map[string]interface{}{
			"strategy": strategyName,
			"score":    avgScore,
			"count":    strategyCount[strategy],
		})

		if avgScore > bestScore {
			bestScore = avgScore
			bestStrategy = strategy
		}
	}

	// 按评分排序
	for i := 0; i < len(strategyRankings); i++ {
		for j := i + 1; j < len(strategyRankings); j++ {
			if strategyRankings[i]["score"].(float64) < strategyRankings[j]["score"].(float64) {
				strategyRankings[i], strategyRankings[j] = strategyRankings[j], strategyRankings[i]
			}
		}
	}

	return map[string]interface{}{
		"recommended_strategy": ais.getStrategyName(bestStrategy),
		"confidence_score":     bestScore,
		"strategy_rankings":    strategyRankings,
		"recommendation_basis": "基于质量(60%)和速度(40%)的综合评分",
	}
}

// assessDataQuality 评估数据质量
func (ais *AdaptiveIndexSelector) assessDataQuality() map[string]interface{} {
	if len(ais.performanceWindow) == 0 {
		return map[string]interface{}{
			"status": "no_data",
		}
	}

	// 检查数据完整性
	validRecords := 0
	for _, record := range ais.performanceWindow {
		if record.Latency > 0 && record.Quality >= 0 && record.Quality <= 1 {
			validRecords++
		}
	}

	completenessRatio := float64(validRecords) / float64(len(ais.performanceWindow))

	// 检查数据新鲜度
	newest := ais.performanceWindow[len(ais.performanceWindow)-1].Timestamp
	oldest := ais.performanceWindow[0].Timestamp
	dataSpan := newest.Sub(oldest)

	// 检查策略多样性
	uniqueStrategies := make(map[IndexStrategy]bool)
	for _, record := range ais.performanceWindow {
		uniqueStrategies[record.Strategy] = true
	}
	strategyDiversity := float64(len(uniqueStrategies))

	qualityLevel := "poor"
	if completenessRatio > 0.95 && strategyDiversity >= 3 {
		qualityLevel = "excellent"
	} else if completenessRatio > 0.8 && strategyDiversity >= 2 {
		qualityLevel = "good"
	} else if completenessRatio > 0.6 {
		qualityLevel = "fair"
	}

	return map[string]interface{}{
		"overall_quality":    qualityLevel,
		"completeness_ratio": completenessRatio,
		"valid_records":      validRecords,
		"total_records":      len(ais.performanceWindow),
		"data_span":          dataSpan.String(),
		"strategy_diversity": len(uniqueStrategies),
		"newest_record":      newest.Format(time.RFC3339),
		"oldest_record":      oldest.Format(time.RFC3339),
	}
}

// detectAnomalies 检测异常
func (ais *AdaptiveIndexSelector) detectAnomalies() map[string]interface{} {
	if len(ais.performanceWindow) < 10 {
		return map[string]interface{}{
			"status": "insufficient_data",
		}
	}

	// 计算延迟和质量的统计信息
	var latencies []float64
	var qualities []float64

	for _, record := range ais.performanceWindow {
		latencies = append(latencies, float64(record.Latency.Milliseconds()))
		qualities = append(qualities, record.Quality)
	}

	// 计算延迟的均值和标准差
	latencyMean := ais.calculateMean(latencies)
	latencyStdDev := ais.calculateStdDev(latencies, latencyMean)

	// 计算质量的均值和标准差
	qualityMean := ais.calculateMean(qualities)
	qualityStdDev := ais.calculateStdDev(qualities, qualityMean)

	// 检测异常值 (使用3-sigma规则)
	var latencyAnomalies []map[string]interface{}
	var qualityAnomalies []map[string]interface{}

	for i, record := range ais.performanceWindow {
		latencyValue := float64(record.Latency.Milliseconds())
		qualityValue := record.Quality

		// 检测延迟异常
		if math.Abs(latencyValue-latencyMean) > 3*latencyStdDev {
			latencyAnomalies = append(latencyAnomalies, map[string]interface{}{
				"index":     i,
				"timestamp": record.Timestamp.Format(time.RFC3339),
				"strategy":  ais.getStrategyName(record.Strategy),
				"value":     record.Latency.String(),
				"deviation": math.Abs(latencyValue-latencyMean) / latencyStdDev,
			})
		}

		// 检测质量异常
		if math.Abs(qualityValue-qualityMean) > 3*qualityStdDev {
			qualityAnomalies = append(qualityAnomalies, map[string]interface{}{
				"index":     i,
				"timestamp": record.Timestamp.Format(time.RFC3339),
				"strategy":  ais.getStrategyName(record.Strategy),
				"value":     qualityValue,
				"deviation": math.Abs(qualityValue-qualityMean) / qualityStdDev,
			})
		}
	}

	return map[string]interface{}{
		"latency_anomalies": latencyAnomalies,
		"quality_anomalies": qualityAnomalies,
		"latency_stats": map[string]interface{}{
			"mean":    latencyMean,
			"std_dev": latencyStdDev,
		},
		"quality_stats": map[string]interface{}{
			"mean":    qualityMean,
			"std_dev": qualityStdDev,
		},
		"total_anomalies": len(latencyAnomalies) + len(qualityAnomalies),
	}
}

// calculateMean 计算均值
func (ais *AdaptiveIndexSelector) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

// calculateStdDev 计算标准差
func (ais *AdaptiveIndexSelector) calculateStdDev(values []float64, mean float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	variance := 0.0
	for _, value := range values {
		diff := value - mean
		variance += diff * diff
	}
	variance /= float64(len(values) - 1)
	return math.Sqrt(variance)
}

// GetOptimalStrategy 根据搜索上下文获取最优策略
func (ais *AdaptiveIndexSelector) GetOptimalStrategy(ctx SearchContext) IndexStrategy {
	ais.mu.RLock()
	defer ais.mu.RUnlock()

	if len(ais.performanceWindow) < 5 {
		return -1 // 数据不足，返回无效策略
	}

	// 根据当前上下文找到最相似的历史记录
	bestStrategy := IndexStrategy(-1)
	bestScore := -1.0

	// 为当前上下文的特征计算权重
	vectorCount := len(ais.db.vectors)
	dimension := len(ctx.QueryVector)

	// 分析各策略在相似上下文下的表现
	strategyPerformance := make(map[IndexStrategy][]float64)

	for _, record := range ais.performanceWindow {
		// 计算上下文相似度
		contextSimilarity := ais.calculateContextSimilarity(ctx, record, vectorCount, dimension)

		// 只考虑相似度较高的记录
		if contextSimilarity > 0.7 {
			// 计算综合性能分数
			latencyScore := 1.0 / (1.0 + float64(record.Latency.Milliseconds())/1000.0)
			performanceScore := 0.6*record.Quality + 0.4*latencyScore
			// 根据上下文相似度加权
			weightedScore := performanceScore * contextSimilarity

			strategyPerformance[record.Strategy] = append(strategyPerformance[record.Strategy], weightedScore)
		}
	}

	// 选择平均性能最好的策略
	for strategy, scores := range strategyPerformance {
		if len(scores) >= 2 { // 至少需要2个样本
			avgScore := ais.calculateMean(scores)
			if avgScore > bestScore {
				bestScore = avgScore
				bestStrategy = strategy
			}
		}
	}

	return bestStrategy
}

// calculateContextSimilarity 计算上下文相似度
func (ais *AdaptiveIndexSelector) calculateContextSimilarity(ctx SearchContext, record PerformanceRecord, currentVectorCount, currentDimension int) float64 {
	// 数据规模相似度
	vectorCountSim := 1.0 - math.Abs(float64(currentVectorCount-record.VectorCount))/math.Max(float64(currentVectorCount), float64(record.VectorCount))

	// 维度相似度
	dimensionSim := 1.0 - math.Abs(float64(currentDimension-record.Dimension))/math.Max(float64(currentDimension), float64(record.Dimension))

	// 时间相似度 (越近的记录相似度越高)
	timeDiff := time.Since(record.Timestamp).Hours()
	timeSim := math.Exp(-timeDiff / 24.0) // 24小时衰减

	// 综合相似度 (加权平均)
	similarity := 0.4*vectorCountSim + 0.3*dimensionSim + 0.3*timeSim

	return math.Max(0, math.Min(1, similarity))
}
