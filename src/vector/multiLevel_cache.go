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
