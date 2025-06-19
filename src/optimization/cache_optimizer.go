package optimization

import (
	"VectorSphere/src/library/logger"
	"VectorSphere/src/vector"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// CacheOptimizer 缓存优化器
type CacheOptimizer struct {
	multiLevelCache *vector.MultiLevelCache
	hitRateMonitor  *HitRateMonitor
	evictionPolicy  *LRUEvictionPolicy
	lastOptimize    time.Time
	mu              sync.RWMutex
}

// HitRateMonitor 命中率监控器
type HitRateMonitor struct {
	totalQueries int64
	cacheHits    int64
	l1Hits       int64
	l2Hits       int64
	l3Hits       int64
}

// NewCacheOptimizer 创建缓存优化器
func NewCacheOptimizer(multiLevelCache *vector.MultiLevelCache) *CacheOptimizer {
	return &CacheOptimizer{
		multiLevelCache: multiLevelCache,
		hitRateMonitor:  NewHitRateMonitor(),
		evictionPolicy:  NewLRUEvictionPolicy(),
		lastOptimize:    time.Now(),
	}
}

// NewHitRateMonitor 创建命中率监控器
func NewHitRateMonitor() *HitRateMonitor {
	return &HitRateMonitor{}
}

// recordHit 记录缓存命中
func (hrm *HitRateMonitor) recordHit(cacheLevel string) {
	atomic.AddInt64(&hrm.totalQueries, 1)
	atomic.AddInt64(&hrm.cacheHits, 1)

	switch cacheLevel {
	case "l1":
		atomic.AddInt64(&hrm.l1Hits, 1)
	case "l2":
		atomic.AddInt64(&hrm.l2Hits, 1)
	case "l3":
		atomic.AddInt64(&hrm.l3Hits, 1)
	default:
		atomic.AddInt64(&hrm.cacheHits, 1)
	}
}

// recordMiss 记录缓存未命中
func (hrm *HitRateMonitor) recordMiss() {
	atomic.AddInt64(&hrm.totalQueries, 1)
}

// GetHitRate 获取命中率
func (hrm *HitRateMonitor) GetHitRate() float64 {
	total := atomic.LoadInt64(&hrm.totalQueries)
	if total == 0 {
		return 0.0
	}

	hits := atomic.LoadInt64(&hrm.cacheHits)
	return float64(hits) / float64(total)
}

// GetL1HitRate 获取L1缓存命中率
func (hrm *HitRateMonitor) GetL1HitRate() float64 {
	total := atomic.LoadInt64(&hrm.totalQueries)
	if total == 0 {
		return 0.0
	}

	hits := atomic.LoadInt64(&hrm.l1Hits)
	return float64(hits) / float64(total)
}

// GetL2HitRate 获取L2缓存命中率
func (hrm *HitRateMonitor) GetL2HitRate() float64 {
	total := atomic.LoadInt64(&hrm.totalQueries)
	if total == 0 {
		return 0.0
	}

	hits := atomic.LoadInt64(&hrm.l2Hits)
	return float64(hits) / float64(total)
}

// GetL3HitRate 获取L3缓存命中率
func (hrm *HitRateMonitor) GetL3HitRate() float64 {
	total := atomic.LoadInt64(&hrm.totalQueries)
	if total == 0 {
		return 0.0
	}

	hits := atomic.LoadInt64(&hrm.l3Hits)
	return float64(hits) / float64(total)
}

// GetStats 获取统计信息
func (hrm *HitRateMonitor) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"total_queries": atomic.LoadInt64(&hrm.totalQueries),
		"cache_hits":    atomic.LoadInt64(&hrm.cacheHits),
		"l1_hits":       atomic.LoadInt64(&hrm.l1Hits),
		"l2_hits":       atomic.LoadInt64(&hrm.l2Hits),
		"l3_hits":       atomic.LoadInt64(&hrm.l3Hits),
		"hit_rate":      hrm.GetHitRate(),
		"l1_hit_rate":   hrm.GetL1HitRate(),
		"l2_hit_rate":   hrm.GetL2HitRate(),
		"l3_hit_rate":   hrm.GetL3HitRate(),
	}
}

// LRUEvictionPolicy LRU淘汰策略
type LRUEvictionPolicy struct {
	accessTimes map[string]time.Time
	mu          sync.RWMutex
}

// NewLRUEvictionPolicy 创建LRU淘汰策略
func NewLRUEvictionPolicy() *LRUEvictionPolicy {
	return &LRUEvictionPolicy{
		accessTimes: make(map[string]time.Time),
	}
}

// RecordAccess 记录访问
func (lru *LRUEvictionPolicy) RecordAccess(key string) {
	lru.mu.Lock()
	defer lru.mu.Unlock()

	lru.accessTimes[key] = time.Now()
}

// ShouldEvict 是否应该淘汰
func (lru *LRUEvictionPolicy) ShouldEvict(key string, timestamp time.Time) bool {
	lru.mu.RLock()
	defer lru.mu.RUnlock()

	lastAccess, exists := lru.accessTimes[key]
	if !exists {
		return true
	}

	// 如果最后访问时间超过1小时，则淘汰
	return time.Since(lastAccess) > time.Hour
}

// GetPriority 获取优先级（越小越优先淘汰）
func (lru *LRUEvictionPolicy) GetPriority(key string) int {
	lru.mu.RLock()
	defer lru.mu.RUnlock()

	lastAccess, exists := lru.accessTimes[key]
	if !exists {
		return 0 // 最低优先级
	}

	// 将时间转换为优先级，最近访问的优先级高
	return int(time.Since(lastAccess).Seconds())
}

// Optimize 优化缓存
func (co *CacheOptimizer) Optimize() error {
	co.mu.Lock()
	defer co.mu.Unlock()

	// 检查是否需要优化
	if time.Since(co.lastOptimize) < 5*time.Minute {
		return nil
	}

	co.lastOptimize = time.Now()
	logger.Info("开始优化缓存...")

	// 获取缓存统计信息
	stats := co.multiLevelCache.GetStats()

	// 根据命中率调整缓存大小
	l1HitRate := float64(stats.L1Hits) / float64(stats.TotalQueries)
	l2HitRate := float64(stats.L2Hits) / float64(stats.TotalQueries)
	l3HitRate := float64(stats.L3Hits) / float64(stats.TotalQueries)

	logger.Info("缓存命中率: L1=%.2f%%, L2=%.2f%%, L3=%.2f%%",
		l1HitRate*100, l2HitRate*100, l3HitRate*100)

	// 根据命中率动态调整缓存大小
	if l1HitRate < 0.5 && l2HitRate > 0.7 {
		// L1命中率低但L2命中率高，增加L1缓存大小
		co.multiLevelCache.IncreaseL1Capacity(2000)
		logger.Info("增加L1缓存容量")
	} else if l1HitRate > 0.9 && l2HitRate < 0.3 {
		// L1命中率高但L2命中率低，减小L1缓存大小，增加L2缓存大小
		co.multiLevelCache.DecreaseL1Capacity(1000)
		co.multiLevelCache.IncreaseL2Capacity(5000)
		logger.Info("减小L1缓存大小，增加L2缓存大小")
	} else if l2HitRate > 0.9 && l3HitRate < 0.3 {
		// L2命中率高但L3命中率低，减小L2缓存大小，增加L3缓存大小
		co.multiLevelCache.DecreaseL2Capacity(2000)
		co.multiLevelCache.IncreaseL3Capacity(10000)
		logger.Info("减小L2缓存大小，增加L3缓存大小")
	} else if l1HitRate < 0.3 && l2HitRate < 0.3 && l3HitRate < 0.3 {
		// 三级缓存命中率都低，清理缓存并重新预热
		co.clearCache()
		co.prewarmCache()
		logger.Info("清理并预热缓存")
	}

	// 执行缓存清理
	co.cleanupCache()

	// 重置统计
	co.resetStats()

	logger.Info("缓存优化完成")
	return nil
}

// cleanupCache 清理缓存
func (co *CacheOptimizer) cleanupCache() {
	// 获取当前时间
	now := time.Now()

	// 清理过期的缓存项
	co.multiLevelCache.CleanupExpired(now.Add(-30 * time.Minute))

	// 清理低命中率的缓存项
	co.multiLevelCache.CleanupLowHitRate(0.1) // 清理命中率低于10%的缓存项

	// 清理超过容量限制的缓存项
	co.multiLevelCache.EnforceCapacityLimits()

	logger.Info("缓存清理完成")
}

// checkCache 检查缓存
func (co *CacheOptimizer) checkCache(key string) (interface{}, bool) {
	// 检查L1缓存
	value, found := co.multiLevelCache.CheckL1Cache(key)
	if found {
		co.hitRateMonitor.recordHit("l1")
		return value, true
	}

	// 检查L2缓存
	value, found = co.multiLevelCache.CheckL2Cache(key)
	if found {
		co.hitRateMonitor.recordHit("l2")
		// 将结果提升到L1缓存
		co.multiLevelCache.PromoteToL1Cache(key, value)
		return value, true
	}

	// 检查L3缓存
	value, found = co.multiLevelCache.CheckL3Cache(key)
	if found {
		co.hitRateMonitor.recordHit("l3")
		// 将结果提升到L2缓存
		co.multiLevelCache.PromoteToL2Cache(key, value)
		return value, true
	}

	// 记录缓存未命中
	co.hitRateMonitor.recordMiss()
	return nil, false
}

// cacheResults 缓存结果
func (co *CacheOptimizer) cacheResults(key string, results []string) {
	// 缓存到多级缓存
	co.multiLevelCache.Put(key, results)

	// 更新LRU策略
	co.evictionPolicy.RecordAccess(key)
}

// generateCacheKey 生成缓存键
func generateCacheKey(vector []float64, k int, options *SearchOptions) string {
	// 生成向量的哈希值
	vectorHash := hashVector(vector)

	// 构建缓存键
	strategy := "auto"
	if options.ForceStrategy != "" {
		strategy = options.ForceStrategy
	}

	return fmt.Sprintf("%s:%d:%s:%.2f:%v:%d",
		vectorHash,
		k,
		strategy,
		options.QualityLevel,
		options.EnableGPU,
		options.Nprobe)
}

// hashVector 哈希向量
func hashVector(vector []float64) string {
	// 简单哈希实现，实际应用中可以使用更复杂的哈希算法
	hash := uint64(0)
	for _, v := range vector {
		// 将浮点数转换为整数进行哈希
		bits := uint64(float64Bits(v))
		hash = hash*31 + bits
	}
	return fmt.Sprintf("%x", hash)
}

// float64Bits 将float64转换为uint64位表示
func float64Bits(f float64) uint64 {
	return *(*uint64)(unsafe.Pointer(&f))
}

// clearCache 清理缓存
func (co *CacheOptimizer) clearCache() {
	// 调用向量数据库清理缓存
	co.multiLevelCache.Clear()
}

// prewarmCache 预热缓存
func (co *CacheOptimizer) prewarmCache() {
	logger.Info("开始预热缓存...")

	// 调用向量数据库系统级预热缓存
	co.multiLevelCache.PrewarmSystem()

	// 预热常用的向量查询模式
	co.prewarmCommonVectorQueries()

	// 预热最近访问的热门查询
	co.prewarmRecentHotQueries()

	// 预热基于用户行为的查询模式
	co.prewarmUserBehaviorQueries()

	logger.Info("缓存预热完成")
}

// prewarmCommonVectorQueries 预热常用的向量查询模式
func (co *CacheOptimizer) prewarmCommonVectorQueries() {
	// 预热常见的向量维度查询
	commonDimensions := []int{128, 256, 512, 768, 1024}
	commonKValues := []int{5, 10, 20, 50, 100}

	for _, dim := range commonDimensions {
		for _, k := range commonKValues {
			// 生成示例向量
			vector := make([]float64, dim)
			for i := range vector {
				vector[i] = float64(i) / float64(dim) // 简单的示例向量
			}

			// 生成缓存键
			options := &SearchOptions{
				QualityLevel: 0.8,
				EnableGPU:    false,
				Nprobe:       10,
			}
			cacheKey := generateCacheKey(vector, k, options)

			// 生成模拟结果
			mockResults := make([]string, k)
			for i := 0; i < k; i++ {
				mockResults[i] = fmt.Sprintf("result_%d_%d_%d", dim, k, i)
			}

			// 预热到缓存
			co.multiLevelCache.Prewarm(cacheKey, mockResults)
		}
	}

	logger.Info("预热了 %d 个常用向量查询模式", len(commonDimensions)*len(commonKValues))
}

// prewarmRecentHotQueries 预热最近访问的热门查询
func (co *CacheOptimizer) prewarmRecentHotQueries() {
	// 从LRU策略中获取最近访问的查询
	co.evictionPolicy.mu.RLock()
	recentQueries := make([]string, 0, 50)
	currentTime := time.Now()

	// 收集最近1小时内访问的查询
	for key, accessTime := range co.evictionPolicy.accessTimes {
		if currentTime.Sub(accessTime) <= time.Hour {
			recentQueries = append(recentQueries, key)
		}
		if len(recentQueries) >= 50 { // 限制数量
			break
		}
	}
	co.evictionPolicy.mu.RUnlock()

	// 为这些查询生成模拟结果并预热
	for i, queryKey := range recentQueries {
		// 生成模拟结果
		mockResults := make([]string, 10) // 默认返回10个结果
		for j := 0; j < 10; j++ {
			mockResults[j] = fmt.Sprintf("hot_result_%d_%d", i, j)
		}

		// 预热到缓存
		co.multiLevelCache.Prewarm(queryKey, mockResults)
	}

	logger.Info("预热了 %d 个最近热门查询", len(recentQueries))
}

// prewarmUserBehaviorQueries 预热基于用户行为的查询模式
func (co *CacheOptimizer) prewarmUserBehaviorQueries() {
	// 基于用户行为模式预热缓存
	userBehaviorPatterns := []struct {
		pattern string
		weight  float64
	}{
		{"similarity_search_text", 0.4},  //文本相似性搜索 (40%权重)
		{"similarity_search_image", 0.3}, //图像相似性搜索 (30%权重)
		{"recommendation_query", 0.2},    //推荐查询 (20%权重)
		{"classification_query", 0.1},    //分类查询 (10%权重)
	}

	for _, pattern := range userBehaviorPatterns {
		// 根据权重决定预热的查询数量
		queryCount := int(pattern.weight * 100)

		for i := 0; i < queryCount; i++ {
			// 生成基于模式的缓存键
			cacheKey := fmt.Sprintf("%s_pattern_%d", pattern.pattern, i)

			// 生成模拟结果
			mockResults := make([]string, 15) // 用户行为查询通常返回更多结果
			for j := 0; j < 15; j++ {
				mockResults[j] = fmt.Sprintf("%s_result_%d", pattern.pattern, j)
			}

			// 预热到缓存
			co.multiLevelCache.Prewarm(cacheKey, mockResults)
		}
	}

	logger.Info("预热了基于用户行为的查询模式")
}

// resetStats 重置统计
func (co *CacheOptimizer) resetStats() {
	atomic.StoreInt64(&co.hitRateMonitor.totalQueries, 0)
	atomic.StoreInt64(&co.hitRateMonitor.cacheHits, 0)
	atomic.StoreInt64(&co.hitRateMonitor.l1Hits, 0)
	atomic.StoreInt64(&co.hitRateMonitor.l2Hits, 0)
	atomic.StoreInt64(&co.hitRateMonitor.l3Hits, 0)
}

// GetHitRateStats 获取命中率统计
func (co *CacheOptimizer) GetHitRateStats() map[string]interface{} {
	return co.hitRateMonitor.GetStats()
}

// OptimizeForWorkload 根据工作负载优化缓存
func (co *CacheOptimizer) OptimizeForWorkload(avgQueriesPerSecond float64, avgResultSize int) {
	// 根据查询率和结果大小优化缓存配置
	if avgQueriesPerSecond > 100 {
		// 高查询率，增加L1缓存大小
		co.multiLevelCache.IncreaseL1Capacity(5000)
		logger.Info("高查询率，增加L1缓存容量")
	}

	if avgResultSize > 1000 {
		// 大结果集，增加L2和L3缓存大小
		co.multiLevelCache.IncreaseL2Capacity(10000)
		co.multiLevelCache.IncreaseL3Capacity(50000)
		logger.Info("大结果集，增加L2和L3缓存容量")
	}
}
