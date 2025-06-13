package vector

import (
	"math"
	"sync"
	"time"
)

// MultiLevelCache 多级缓存结构
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

// queryCache 查询缓存结构
type queryCache struct {
	results   []string
	timestamp int64
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

// GetStats 获取缓存统计信息
func (c *MultiLevelCache) GetStats() CacheStats {
	c.statsMu.RLock()
	defer c.statsMu.RUnlock()
	return c.stats
}

// IncreaseL1Capacity 增加L1缓存容量
func (c *MultiLevelCache) IncreaseL1Capacity(increment int) {
	c.l1Mu.Lock()
	defer c.l1Mu.Unlock()
	c.l1Capacity += increment
}

// DecreaseL1Capacity 减小L1缓存容量
func (c *MultiLevelCache) DecreaseL1Capacity(decrement int) {
	c.l1Mu.Lock()
	defer c.l1Mu.Unlock()
	// 确保容量不会小于最小值
	if c.l1Capacity > decrement {
		c.l1Capacity -= decrement
		// 如果当前缓存项数量超过新容量，移除最旧的项
		c.enforceL1CapacityLimit()
	}
}

// IncreaseL2Capacity 增加L2缓存容量
func (c *MultiLevelCache) IncreaseL2Capacity(increment int) {
	c.l2Mu.Lock()
	defer c.l2Mu.Unlock()
	c.l2Capacity += increment
}

// DecreaseL2Capacity 减小L2缓存容量
func (c *MultiLevelCache) DecreaseL2Capacity(decrement int) {
	c.l2Mu.Lock()
	defer c.l2Mu.Unlock()
	// 确保容量不会小于最小值
	if c.l2Capacity > decrement {
		c.l2Capacity -= decrement
		// 如果当前缓存项数量超过新容量，移除最旧的项
		c.enforceL2CapacityLimit()
	}
}

// IncreaseL3Capacity 增加L3缓存容量
func (c *MultiLevelCache) IncreaseL3Capacity(increment int) {
	// L3缓存是基于磁盘的，这里可能需要调整配置或分配更多磁盘空间
	// 简化实现，仅记录日志
}

// CleanupExpired 清理过期的缓存项
func (c *MultiLevelCache) CleanupExpired(expiryTime time.Time) {
	timestamp := expiryTime.Unix()

	// 清理L1缓存
	c.l1Mu.Lock()
	for k, v := range c.l1Cache {
		if v.timestamp < timestamp {
			delete(c.l1Cache, k)
		}
	}
	c.l1Mu.Unlock()

	// 清理L2缓存
	c.l2Mu.Lock()
	for k, v := range c.l2Cache {
		if v.timestamp < timestamp {
			delete(c.l2Cache, k)
		}
	}
	c.l2Mu.Unlock()

	// 清理L3缓存
	// 这里需要实现清理磁盘缓存的逻辑
	// ...
}

// CleanupLowHitRate 清理低命中率的缓存项
func (c *MultiLevelCache) CleanupLowHitRate(minHitRate float64) {
	// 这里需要实现根据命中率清理缓存的逻辑
	// 由于当前实现没有跟踪每个缓存项的命中率，这里只是一个占位实现
	// 实际实现可能需要额外的数据结构来跟踪每个缓存项的访问次数
}

// EnforceCapacityLimits 强制执行容量限制
func (c *MultiLevelCache) EnforceCapacityLimits() {
	// 强制执行L1缓存容量限制
	c.l1Mu.Lock()
	c.enforceL1CapacityLimit()
	c.l1Mu.Unlock()

	// 强制执行L2缓存容量限制
	c.l2Mu.Lock()
	c.enforceL2CapacityLimit()
	c.l2Mu.Unlock()

	// L3缓存容量限制可能需要特殊处理
	// ...
}

// enforceL1CapacityLimit 强制执行L1缓存容量限制的内部方法
func (c *MultiLevelCache) enforceL1CapacityLimit() {
	for len(c.l1Cache) > c.l1Capacity {
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
}

// enforceL2CapacityLimit 强制执行L2缓存容量限制的内部方法
func (c *MultiLevelCache) enforceL2CapacityLimit() {
	for len(c.l2Cache) > c.l2Capacity {
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
}

// CheckL1Cache 检查L1缓存
func (c *MultiLevelCache) CheckL1Cache(key string) (interface{}, bool) {
	c.l1Mu.RLock()
	defer c.l1Mu.RUnlock()

	if cache, found := c.l1Cache[key]; found && time.Now().Unix()-cache.timestamp < 300 {
		c.statsMu.Lock()
		c.stats.L1Hits++
		c.statsMu.Unlock()
		return cache.results, true
	}

	return nil, false
}

// CheckL2Cache 检查L2缓存
func (c *MultiLevelCache) CheckL2Cache(key string) (interface{}, bool) {
	c.l2Mu.RLock()
	defer c.l2Mu.RUnlock()

	if cache, found := c.l2Cache[key]; found && time.Now().Unix()-cache.timestamp < 1800 {
		c.statsMu.Lock()
		c.stats.L2Hits++
		c.statsMu.Unlock()
		return cache.results, true
	}

	return nil, false
}

// CheckL3Cache 检查L3缓存
func (c *MultiLevelCache) CheckL3Cache(key string) (interface{}, bool) {
	// 这里需要实现从磁盘读取缓存的逻辑
	// 简化实现，始终返回未命中
	return nil, false
}

// PromoteToL1Cache 将结果提升到L1缓存
func (c *MultiLevelCache) PromoteToL1Cache(key string, value interface{}) {
	c.l1Mu.Lock()
	defer c.l1Mu.Unlock()

	// 将结果转换为queryCache类型
	results, ok := value.([]string)
	if !ok {
		// 如果类型转换失败，尝试其他可能的类型
		return
	}

	cache := queryCache{
		results:   results,
		timestamp: time.Now().Unix(),
	}

	c.l1Cache[key] = cache

	// 如果L1缓存超出容量，移除最旧的项
	c.enforceL1CapacityLimit()
}

// PromoteToL2Cache 将结果提升到L2缓存
func (c *MultiLevelCache) PromoteToL2Cache(key string, value interface{}) {
	c.l2Mu.Lock()
	defer c.l2Mu.Unlock()

	// 将结果转换为queryCache类型
	results, ok := value.([]string)
	if !ok {
		// 如果类型转换失败，尝试其他可能的类型
		return
	}

	cache := queryCache{
		results:   results,
		timestamp: time.Now().Unix(),
	}

	c.l2Cache[key] = cache

	// 如果L2缓存超出容量，移除最旧的项
	c.enforceL2CapacityLimit()
}

// Clear 清空缓存
func (c *MultiLevelCache) Clear() {
	// 清空L1缓存
	c.l1Mu.Lock()
	c.l1Cache = make(map[string]queryCache, c.l1Capacity)
	c.l1Mu.Unlock()

	// 清空L2缓存
	c.l2Mu.Lock()
	c.l2Cache = make(map[string]queryCache, c.l2Capacity)
	c.l2Mu.Unlock()

	// 清空L3缓存
	// 这里需要实现清空磁盘缓存的逻辑
	// ...

	// 重置统计信息
	c.statsMu.Lock()
	c.stats = CacheStats{}
	c.statsMu.Unlock()
}

// Prewarm 预热缓存
func (c *MultiLevelCache) Prewarm() {
	// 这里实现预热缓存的逻辑
	// 例如，可以加载常用的查询结果到缓存中

	// 预分配内存以提高性能
	c.l1Mu.Lock()
	c.l1Cache = make(map[string]queryCache, c.l1Capacity)
	c.l1Mu.Unlock()

	c.l2Mu.Lock()
	c.l2Cache = make(map[string]queryCache, c.l2Capacity)
	c.l2Mu.Unlock()

	// 重置统计信息
	c.statsMu.Lock()
	c.stats = CacheStats{}
	c.statsMu.Unlock()

	// 注意：实际生产环境中，可以从历史查询日志中加载热门查询
	// 或者从预定义的热门查询列表中加载数据到缓存中
}
