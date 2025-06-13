package optimization

import (
	"VectorSphere/src/library/log"
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
	log.Info("开始优化缓存...")

	// 获取缓存统计信息
	stats := co.multiLevelCache.GetStats()

	// 根据命中率调整缓存大小
	l1HitRate := float64(stats.L1Hits) / float64(stats.TotalQueries)
	l2HitRate := float64(stats.L2Hits) / float64(stats.TotalQueries)
	l3HitRate := float64(stats.L3Hits) / float64(stats.TotalQueries)

	log.Info("缓存命中率: L1=%.2f%%, L2=%.2f%%, L3=%.2f%%",
		l1HitRate*100, l2HitRate*100, l3HitRate*100)

	// 根据命中率动态调整缓存大小
	if l1HitRate < 0.5 && l2HitRate > 0.7 {
		// L1命中率低但L2命中率高，增加L1缓存大小
		co.multiLevelCache.IncreaseL1Capacity(2000)
		log.Info("增加L1缓存容量")
	} else if l1HitRate > 0.9 && l2HitRate < 0.3 {
		// L1命中率高但L2命中率低，减小L1缓存大小，增加L2缓存大小
		co.multiLevelCache.DecreaseL1Capacity(1000)
		co.multiLevelCache.IncreaseL2Capacity(5000)
		log.Info("减小L1缓存大小，增加L2缓存大小")
	} else if l2HitRate > 0.9 && l3HitRate < 0.3 {
		// L2命中率高但L3命中率低，减小L2缓存大小，增加L3缓存大小
		co.multiLevelCache.DecreaseL2Capacity(2000)
		co.multiLevelCache.IncreaseL3Capacity(10000)
		log.Info("减小L2缓存大小，增加L3缓存大小")
	} else if l1HitRate < 0.3 && l2HitRate < 0.3 && l3HitRate < 0.3 {
		// 三级缓存命中率都低，清理缓存并重新预热
		co.clearCache()
		co.prewarmCache()
		log.Info("清理并预热缓存")
	}

	// 执行缓存清理
	co.cleanupCache()

	// 重置统计
	co.resetStats()

	log.Info("缓存优化完成")
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

	log.Info("缓存清理完成")
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
func (co *CacheOptimizer) cacheResults(key string, value interface{}) {
	// 缓存到多级缓存
	co.multiLevelCache.Put(key, value)

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
	// 调用向量数据库预热缓存
	co.multiLevelCache.Prewarm()
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
		log.Info("高查询率，增加L1缓存容量")
	}

	if avgResultSize > 1000 {
		// 大结果集，增加L2和L3缓存大小
		co.multiLevelCache.IncreaseL2Capacity(10000)
		co.multiLevelCache.IncreaseL3Capacity(50000)
		log.Info("大结果集，增加L2和L3缓存容量")
	}
}
