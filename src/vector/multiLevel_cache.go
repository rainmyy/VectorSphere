package vector

import (
	"VectorSphere/src/library/logger"
	"VectorSphere/src/library/storage"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// HitRateStats 命中率统计结构
type HitRateStats struct {
	HitCount    int64 // 命中次数
	AccessCount int64 // 访问次数
	LastAccess  int64 // 最后访问时间戳
}

// MultiLevelCache 多级缓存结构
type MultiLevelCache struct {
	// L1: 内存缓存 - 最快，容量小
	l1Cache    map[string]queryCache
	l1Capacity int
	l1Mu       sync.RWMutex

	// L2: 共享内存缓存 - 较快，容量中等
	l2Capacity      int
	l2Mu            sync.RWMutex
	l2SharedMem     *storage.SharedMemory // 共享内存管理器
	l2FallbackCache map[string]queryCache // 普通内存fallback缓存

	// L3: 磁盘缓存 - 较慢，容量大
	l3CachePath string
	l3Capacity  int
	l3FileExt   string
	l3TTL       time.Duration
	l3Mu        sync.RWMutex

	// 缓存统计
	stats   CacheStats
	statsMu sync.RWMutex

	// 命中率跟踪
	hitRateStats map[string]*HitRateStats // 每个缓存项的命中率统计
	hitRateMu    sync.RWMutex             // 命中率统计的读写锁
}

// CacheDataType 缓存数据类型枚举
type CacheDataType int

const (
	TypeUnknown      CacheDataType = iota
	TypeStringArray                // []string 类型
	TypeFloat64Array               // []float64 类型
	TypeIntArray                   // []int 类型
	TypeMap                        // map[string]interface{} 类型
	TypeStruct                     // 自定义结构体类型
	TypeBytes                      // []byte 类型
)

// queryCache 查询缓存结构
type queryCache struct {
	Results   interface{}   `json:"results"`
	Timestamp int64         `json:"timestamp"`
	DataType  CacheDataType `json:"data_type"`
}

// CacheStats 缓存统计信息
type CacheStats struct {
	L1Hits       int64
	L2Hits       int64
	L3Hits       int64
	TotalQueries int64
}

// NewMultiLevelCache 创建新的多级缓存
func NewMultiLevelCache(l1Capacity, l2Capacity, l3Capacity int, l3Path string) *MultiLevelCache {
	// 清理和验证L3缓存路径
	if l3Path != "" {
		// 清理路径中的特殊字符
		l3Path = filepath.Clean(l3Path)

		// 确保L3缓存目录存在
		err := os.MkdirAll(l3Path, 0755)
		if err != nil {
			// 如果创建失败，记录错误但不返回nil，而是禁用L3缓存
			fmt.Printf("Warning: Failed to create L3 cache directory %s: %v. L3 cache will be disabled.\n", l3Path, err)
			l3Path = "" // 禁用L3缓存
		}
	}

	cache := &MultiLevelCache{
		l1Cache:      make(map[string]queryCache, l1Capacity),
		l1Capacity:   l1Capacity,
		l2Capacity:   l2Capacity,
		l3CachePath:  l3Path,
		l3Capacity:   l3Capacity,
		l3FileExt:    ".cache",
		l3TTL:        24 * time.Hour, // 默认L3缓存过期时间为24小时
		stats:        CacheStats{},
		hitRateStats: make(map[string]*HitRateStats),
	}

	// 初始化共享内存
	sharedMem, err := storage.NewSharedMemory()
	if err != nil {
		// 如果共享内存初始化失败，记录错误并回退到使用普通map
		fmt.Printf("初始化共享内存失败: %v，将使用普通内存作为L2缓存\n", err)
	}
	cache.l2SharedMem = sharedMem

	return cache
}

// updateHitRateStats 更新命中率统计
func (c *MultiLevelCache) updateHitRateStats(key string, hit bool) {
	c.hitRateMu.Lock()
	defer c.hitRateMu.Unlock()

	if c.hitRateStats[key] == nil {
		c.hitRateStats[key] = &HitRateStats{}
	}

	stats := c.hitRateStats[key]
	stats.AccessCount++
	stats.LastAccess = time.Now().Unix()

	if hit {
		stats.HitCount++
	}
}

// Get 从多级缓存获取结果
func (c *MultiLevelCache) Get(key string) (interface{}, bool) {
	// 更新查询计数
	c.statsMu.Lock()
	c.stats.TotalQueries++
	c.statsMu.Unlock()

	// 尝试从 L1 缓存获取
	c.l1Mu.RLock()
	if cache, found := c.l1Cache[key]; found && time.Now().Unix()-cache.Timestamp < 300 {
		c.l1Mu.RUnlock()
		c.statsMu.Lock()
		c.stats.L1Hits++
		c.statsMu.Unlock()
		// 更新命中率统计
		c.updateHitRateStats(key, true)
		return cache.Results, true
	}
	c.l1Mu.RUnlock()

	// 尝试从 L2 缓存获取
	c.l2Mu.RLock()
	if c.l2SharedMem != nil {
		// 使用共享内存
		rawResults, timestamp, found := c.l2SharedMem.Get(key)
		if found && time.Now().Unix()-timestamp < 1800 {
			// 处理不同类型的数据
			var results interface{} = rawResults
			var dataType = TypeStringArray // 默认为字符串数组类型

			// 检查是否为新格式（包含类型信息的序列化数据）
			// rawResults已经是[]string类型，不需要类型断言
			if len(rawResults) == 2 {
				// 尝试解析数据类型
				typeID := 0
				_, err := fmt.Sscanf(rawResults[1], "%d", &typeID)
				if err == nil {
					dataType = CacheDataType(typeID)

					// 根据数据类型反序列化
					switch dataType {
					case TypeStringArray:
						// 尝试解析为字符串数组
						var strResults []string
						err := json.Unmarshal([]byte(rawResults[0]), &strResults)
						if err == nil {
							results = strResults
						} else {
							results = rawResults
						}
					case TypeFloat64Array:
						// 尝试解析为float64数组
						var floatResults []float64
						err := json.Unmarshal([]byte(rawResults[0]), &floatResults)
						if err == nil {
							results = floatResults
						}
					case TypeIntArray:
						// 尝试解析为int数组
						var intResults []int
						err := json.Unmarshal([]byte(rawResults[0]), &intResults)
						if err == nil {
							results = intResults
						}
					case TypeMap:
						// 尝试解析为map
						var mapResults map[string]interface{}
						err := json.Unmarshal([]byte(rawResults[0]), &mapResults)
						if err == nil {
							results = mapResults
						}
					case TypeBytes:
						// 尝试解析为字节数组
						results = []byte(rawResults[0])
					default:
						// 对于未知类型或结构体类型，保留原始JSON字符串
						results = rawResults[0]
					}
				}
			}

			// 构造缓存对象
			cache := queryCache{
				Results:   results,
				Timestamp: timestamp,
				DataType:  dataType,
			}
			// 将结果提升到 L1 缓存
			c.l1Mu.Lock()
			c.l1Cache[key] = cache
			// 如果 L1 缓存超出容量，移除最旧的项
			if len(c.l1Cache) > c.l1Capacity {
				var oldestKey string
				var oldestTime int64 = math.MaxInt64
				for k, v := range c.l1Cache {
					if v.Timestamp < oldestTime {
						oldestTime = v.Timestamp
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
			// 更新命中率统计
			c.updateHitRateStats(key, true)
			return cache.Results, true
		}
	} else if c.l2FallbackCache != nil {
		// 使用fallback缓存
		if cache, found := c.l2FallbackCache[key]; found && time.Now().Unix()-cache.Timestamp < 1800 {
			// 将结果提升到 L1 缓存
			c.l1Mu.Lock()
			c.l1Cache[key] = cache
			// 如果 L1 缓存超出容量，移除最旧的项
			if len(c.l1Cache) > c.l1Capacity {
				var oldestKey string
				var oldestTime int64 = math.MaxInt64
				for k, v := range c.l1Cache {
					if v.Timestamp < oldestTime {
						oldestTime = v.Timestamp
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
			// 更新命中率统计
			c.updateHitRateStats(key, true)
			return cache.Results, true
		}
	}
	c.l2Mu.RUnlock()

	// 尝试从 L3 缓存获取
	c.l3Mu.RLock()
	cache, found := c.readFromL3Cache(key)
	c.l3Mu.RUnlock()

	if found {
		// 将结果提升到 L1 和 L2 缓存
		go func() {
			// 提升到 L2 缓存
			c.l2Mu.Lock()

			if c.l2SharedMem != nil {
				// 根据数据类型处理L2缓存存储
				var err error
				switch cache.DataType {
				case TypeStringArray:
					// 字符串数组类型，直接使用共享内存的Put方法
					if strArray, ok := cache.Results.([]string); ok {
						err = c.l2SharedMem.Put(key, strArray, cache.Timestamp)
					} else {
						err = fmt.Errorf("类型断言失败：预期[]string，实际%T", cache.Results)
					}
				default:
					// 其他类型需要先序列化为JSON字符串，然后存储
					data, jsonErr := json.Marshal(cache.Results)
					if jsonErr != nil {
						err = fmt.Errorf("序列化数据失败: %v", jsonErr)
					} else {
						// 将序列化后的数据作为字符串数组存储
						strData := []string{string(data), fmt.Sprintf("%d", cache.DataType)}
						err = c.l2SharedMem.Put(key, strData, cache.Timestamp)
					}
				}

				if err != nil {
					// 记录错误，但不中断程序
					fmt.Printf("Error writing to L2 shared memory: %v\n", err)
				}

				// 检查共享内存中的条目数量是否超过容量限制
				if c.l2SharedMem.Count() > c.l2Capacity {
					// 获取所有键
					keys := c.l2SharedMem.GetAllKeys()

					// 如果键数量超过容量，删除一些旧的键
					// 注意：这里简化处理，实际应该按时间戳排序
					if len(keys) > c.l2Capacity {
						// 删除超出部分的键
						for i := 0; i < len(keys)-c.l2Capacity; i++ {
							c.l2SharedMem.Delete(keys[i])
						}
					}
				}
			}

			c.l2Mu.Unlock()

			// 提升到 L1 缓存
			c.l1Mu.Lock()
			c.l1Cache[key] = cache
			// 如果 L1 缓存超出容量，移除最旧的项
			if len(c.l1Cache) > c.l1Capacity {
				var oldestKey string
				var oldestTime int64 = math.MaxInt64
				for k, v := range c.l1Cache {
					if v.Timestamp < oldestTime {
						oldestTime = v.Timestamp
						oldestKey = k
					}
				}
				delete(c.l1Cache, oldestKey)
			}
			c.l1Mu.Unlock()
		}()

		c.statsMu.Lock()
		c.stats.L3Hits++
		c.statsMu.Unlock()
		// 更新命中率统计
		c.updateHitRateStats(key, true)
		return cache.Results, true
	}

	// 所有缓存层都未命中，记录未命中
	c.updateHitRateStats(key, false)
	return nil, false
}

// Put 将结果存入多级缓存
func (c *MultiLevelCache) Put(key string, results interface{}) {
	// 检测数据类型
	dataType := c.detectDataType(results)

	cache := queryCache{
		Results:   results,
		Timestamp: time.Now().Unix(),
		DataType:  dataType,
	}

	// 存入 L1 缓存
	c.l1Mu.Lock()
	// 如果键已存在，直接更新
	if _, exists := c.l1Cache[key]; !exists {
		// 如果添加新项会超出容量，先移除最旧的项
		for len(c.l1Cache) >= c.l1Capacity {
			var oldestKey string
			var oldestTime int64 = math.MaxInt64
			for k, v := range c.l1Cache {
				if v.Timestamp < oldestTime {
					oldestTime = v.Timestamp
					oldestKey = k
				}
			}
			if oldestKey != "" {
				delete(c.l1Cache, oldestKey)
			} else {
				break
			}
		}
	}
	c.l1Cache[key] = cache
	c.l1Mu.Unlock()

	// 异步存入 L2 和 L3 缓存
	go func() {
		// 存入 L2 缓存
		c.l2Mu.Lock()
		if c.l2SharedMem != nil {
			// 根据数据类型处理L2缓存存储
			var err error
			switch cache.DataType {
			case TypeStringArray:
				// 字符串数组类型，直接使用共享内存的Put方法
				if strArray, ok := cache.Results.([]string); ok {
					err = c.l2SharedMem.Put(key, strArray, cache.Timestamp)
				} else {
					err = fmt.Errorf("类型断言失败：预期[]string，实际%T", cache.Results)
				}
			default:
				// 其他类型需要先序列化为JSON字符串，然后存储
				data, jsonErr := json.Marshal(cache.Results)
				if jsonErr != nil {
					err = fmt.Errorf("序列化数据失败: %v", jsonErr)
				} else {
					// 将序列化后的数据作为字符串数组存储
					strData := []string{string(data), fmt.Sprintf("%d", cache.DataType)}
					err = c.l2SharedMem.Put(key, strData, cache.Timestamp)
				}
			}

			if err != nil {
				// 记录错误，但不中断程序
				fmt.Printf("Error writing to L2 shared memory: %v\n", err)
			}

			// 检查共享内存中的条目数量是否超过容量限制
			if c.l2SharedMem.Count() > c.l2Capacity {
				// 获取所有键
				keys := c.l2SharedMem.GetAllKeys()

				// 如果键数量超过容量，删除一些旧的键
				// 注意：这里简化处理，实际应该按时间戳排序
				if len(keys) > c.l2Capacity {
					// 删除超出部分的键
					for i := 0; i < len(keys)-c.l2Capacity; i++ {
						c.l2SharedMem.Delete(keys[i])
					}
				}
			}
		} else {
			// 如果共享内存不可用，使用普通内存作为fallback
			if c.l2FallbackCache == nil {
				c.l2FallbackCache = make(map[string]queryCache)
			}

			// 存储到fallback缓存
			c.l2FallbackCache[key] = cache

			// 检查并执行容量限制
			if len(c.l2FallbackCache) > c.l2Capacity {
				// 按时间戳排序，删除最旧的项
				type cacheItem struct {
					key       string
					timestamp int64
				}

				var items []cacheItem
				for k, v := range c.l2FallbackCache {
					items = append(items, cacheItem{key: k, timestamp: v.Timestamp})
				}

				// 按时间戳排序（最旧的在前面）
				sort.Slice(items, func(i, j int) bool {
					return items[i].timestamp < items[j].timestamp
				})

				// 删除超出容量的最旧项
				deleteCount := len(c.l2FallbackCache) - c.l2Capacity
				for i := 0; i < deleteCount; i++ {
					delete(c.l2FallbackCache, items[i].key)
				}
			}
		}
		c.l2Mu.Unlock()

		// 存入 L3 缓存
		c.l3Mu.Lock()
		err := c.writeToL3Cache(key, cache)
		c.l3Mu.Unlock()
		if err != nil {
			// 记录错误，但不中断程序
			fmt.Printf("Error writing to L3 cache: %v\n", err)
		}
	}()
}

// GetStats 获取缓存统计信息
func (c *MultiLevelCache) GetStats() CacheStats {
	c.statsMu.RLock()
	defer c.statsMu.RUnlock()
	return c.stats
}

// GetHitRateStats 获取指定键的命中率统计信息
func (c *MultiLevelCache) GetHitRateStats(key string) (float64, int64, int64, bool) {
	c.hitRateMu.RLock()
	defer c.hitRateMu.RUnlock()

	stats, exists := c.hitRateStats[key]
	if !exists {
		return 0.0, 0, 0, false
	}

	hitRate := 0.0
	if stats.AccessCount > 0 {
		hitRate = float64(stats.HitCount) / float64(stats.AccessCount)
	}

	return hitRate, stats.HitCount, stats.AccessCount, true
}

// GetAllHitRateStats 获取所有缓存项的命中率统计信息
func (c *MultiLevelCache) GetAllHitRateStats() map[string]float64 {
	c.hitRateMu.RLock()
	defer c.hitRateMu.RUnlock()

	result := make(map[string]float64)
	for key, stats := range c.hitRateStats {
		if stats.AccessCount > 0 {
			result[key] = float64(stats.HitCount) / float64(stats.AccessCount)
		} else {
			result[key] = 0.0
		}
	}

	return result
}

// ResetHitRateStats 重置所有命中率统计信息
func (c *MultiLevelCache) ResetHitRateStats() {
	c.hitRateMu.Lock()
	defer c.hitRateMu.Unlock()

	// 清空所有命中率统计
	c.hitRateStats = make(map[string]*HitRateStats)
}

// ResetHitRateStatsForKey 重置指定键的命中率统计信息
func (c *MultiLevelCache) ResetHitRateStatsForKey(key string) {
	c.hitRateMu.Lock()
	defer c.hitRateMu.Unlock()

	delete(c.hitRateStats, key)
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
	// L3缓存是基于磁盘的，需要检查磁盘空间并调整配置
	if increment <= 0 {
		fmt.Printf("Warning: Invalid L3 capacity increment: %d\n", increment)
		return
	}

	// 检查L3缓存路径是否可用
	if c.l3CachePath == "" {
		fmt.Printf("Warning: L3 cache path is not configured, cannot increase capacity\n")
		return
	}

	// 获取当前磁盘使用情况
	c.l3Mu.Lock()
	defer c.l3Mu.Unlock()

	// 检查磁盘可用空间
	availableSpace, err := c.getAvailableDiskSpace(c.l3CachePath)
	if err != nil {
		fmt.Printf("Error checking disk space for L3 cache: %v\n", err)
		return
	}

	// 估算增加容量所需的磁盘空间（假设每个缓存项平均1KB）
	estimatedSpaceNeeded := int64(increment * 1024) // 1KB per cache item

	if availableSpace < estimatedSpaceNeeded {
		fmt.Printf("Warning: Insufficient disk space. Available: %d bytes, Required: %d bytes\n",
			availableSpace, estimatedSpaceNeeded)
		// 根据可用空间调整增量
		adjustedIncrement := int(availableSpace / 1024)
		if adjustedIncrement > 0 {
			fmt.Printf("Adjusting L3 capacity increment to %d based on available disk space\n", adjustedIncrement)
			increment = adjustedIncrement
		} else {
			fmt.Printf("Error: No sufficient disk space available for L3 cache expansion\n")
			return
		}
	}

	// 更新L3缓存容量
	oldCapacity := c.l3Capacity
	c.l3Capacity += increment

	// 记录容量变更日志
	fmt.Printf("L3 cache capacity increased from %d to %d (increment: %d)\n",
		oldCapacity, c.l3Capacity, increment)

	// 创建一个标记文件记录容量配置
	c.updateL3CapacityConfig()
}

// CleanupExpired 清理过期的缓存项
func (c *MultiLevelCache) CleanupExpired(expiryTime time.Time) {
	timestamp := expiryTime.Unix()

	// 清理L1缓存 - 删除时间戳早于过期时间的项
	c.l1Mu.Lock()
	for k, v := range c.l1Cache {
		if v.Timestamp > timestamp {
			delete(c.l1Cache, k)
		}
	}
	c.l1Mu.Unlock()

	// 清理L2缓存
	c.l2Mu.Lock()
	if c.l2SharedMem != nil {
		// 获取所有键
		keys := c.l2SharedMem.GetAllKeys()

		// 检查每个键对应的条目是否过期
		for _, key := range keys {
			_, entryTimestamp, found := c.l2SharedMem.Get(key)
			if found && entryTimestamp > timestamp {
				c.l2SharedMem.Delete(key)
			}
		}
	} else if c.l2FallbackCache != nil {
		// 清理fallback缓存中的过期项
		for k, v := range c.l2FallbackCache {
			if v.Timestamp > timestamp {
				delete(c.l2FallbackCache, k)
			}
		}
	}
	c.l2Mu.Unlock()

	// 清理L3缓存
	if c.l3CachePath != "" {
		go func() {
			c.l3Mu.Lock()
			defer c.l3Mu.Unlock()

			// 遍历L3缓存目录
			files, err := os.ReadDir(c.l3CachePath)
			if err != nil {
				fmt.Printf("Error reading L3 cache directory: %v\n", err)
				return
			}

			for _, file := range files {
				// 只处理缓存文件
				if !strings.HasSuffix(file.Name(), c.l3FileExt) {
					continue
				}

				filePath := filepath.Join(c.l3CachePath, file.Name())

				// 读取文件内容
				data, err := os.ReadFile(filePath)
				if err != nil {
					continue
				}

				// 反序列化
				cache, err := c.deserializeCache(data)
				if err != nil {
					// 无法解析的文件直接删除
					err := os.Remove(filePath)
					if err != nil {
						return
					}
					continue
				}

				// 检查是否过期
				if cache.Timestamp > timestamp {
					err := os.Remove(filePath)
					if err != nil {
						return
					}
				}
			}
		}()
	}

	// 清理过期的命中率统计信息
	c.hitRateMu.Lock()
	for key, stats := range c.hitRateStats {
		// 如果最后访问时间早于过期时间，删除统计信息
		if stats.LastAccess > timestamp {
			delete(c.hitRateStats, key)
		}
	}
	c.hitRateMu.Unlock()
}

// CleanupLowHitRate 清理低命中率的缓存项
func (c *MultiLevelCache) CleanupLowHitRate(minHitRate float64) {
	c.hitRateMu.Lock()
	defer c.hitRateMu.Unlock()

	// 收集需要删除的低命中率缓存项
	var keysToDelete []string
	currentTime := time.Now().Unix()

	for key, stats := range c.hitRateStats {
		// 跳过访问次数太少的项（至少需要10次访问才有统计意义）
		if stats.AccessCount < 10 {
			continue
		}

		// 计算命中率
		hitRate := float64(stats.HitCount) / float64(stats.AccessCount)

		// 如果命中率低于阈值，或者长时间未访问（超过1小时），标记为删除
		if hitRate < minHitRate || (currentTime-stats.LastAccess) > 3600 {
			keysToDelete = append(keysToDelete, key)
		}
	}

	// 从各级缓存中删除低命中率的项
	for _, key := range keysToDelete {
		// 从L1缓存删除
		c.l1Mu.Lock()
		delete(c.l1Cache, key)
		c.l1Mu.Unlock()

		// 从L2缓存删除
		c.l2Mu.Lock()
		if c.l2SharedMem != nil {
			c.l2SharedMem.Delete(key)
		} else if c.l2FallbackCache != nil {
			delete(c.l2FallbackCache, key)
		}
		c.l2Mu.Unlock()

		// 从L3缓存删除
		c.l3Mu.Lock()
		c.deleteFromL3Cache(key)
		c.l3Mu.Unlock()

		// 删除命中率统计
		delete(c.hitRateStats, key)
	}

	// 记录清理信息
	if len(keysToDelete) > 0 {
		fmt.Printf("Cleaned up %d low hit rate cache items (min hit rate: %.2f)\n", len(keysToDelete), minHitRate)
	}
}

// EnforceCapacityLimits 强制执行容量限制
func (c *MultiLevelCache) EnforceCapacityLimits() {
	// 强制执行L1缓存容量限制
	c.l1Mu.Lock()
	c.enforceL1CapacityLimit()
	c.l1Mu.Unlock()

	// 强制执行L2缓存容量限制
	c.l2Mu.Lock()
	if c.l2SharedMem != nil {
		// 检查共享内存中的条目数量是否超过容量限制
		if c.l2SharedMem.Count() > c.l2Capacity {
			// 获取所有键
			keys := c.l2SharedMem.GetAllKeys()

			// 如果键数量超过容量，删除一些旧的键
			// 注意：这里简化处理，实际应该按时间戳排序
			if len(keys) > c.l2Capacity {
				// 删除超出部分的键
				for i := 0; i < len(keys)-c.l2Capacity; i++ {
					c.l2SharedMem.Delete(keys[i])
				}
			}
		}
	}
	c.l2Mu.Unlock()

	// 强制执行L3缓存容量限制
	if c.l3CachePath != "" && c.l3Capacity > 0 {
		go func() {
			c.l3Mu.Lock()
			defer c.l3Mu.Unlock()

			// 获取所有缓存文件
			files, err := fs.ReadDir(os.DirFS(c.l3CachePath), c.l3CachePath)
			if err != nil {
				fmt.Printf("Error reading L3 cache directory: %v\n", err)
				return
			}

			// 过滤出缓存文件并获取其信息
			type cacheFileInfo struct {
				path      string
				timestamp int64
				size      int64
			}
			var cacheFiles []cacheFileInfo
			totalSize := int64(0)

			for _, file := range files {
				if !strings.HasSuffix(file.Name(), c.l3FileExt) {
					continue
				}

				filePath := filepath.Join(c.l3CachePath, file.Name())
				var fileInfo fs.FileInfo
				fileInfo, err = file.Info()
				if err != nil {
					logger.Error("failed to retrieve file information", err)
					continue
				}
				fileSize := fileInfo.Size()
				totalSize += fileSize

				// 读取文件内容获取时间戳
				data, err := fs.ReadFile(os.DirFS(filePath), filePath)
				if err != nil {
					logger.Error("failed to retrieve file information", err)
					continue
				}

				// 反序列化
				cache, err := c.deserializeCache(data)
				if err != nil {
					// 无法解析的文件直接删除
					err := os.Remove(filePath)
					if err != nil {
						return
					}
					continue
				}

				cacheFiles = append(cacheFiles, cacheFileInfo{
					path:      filePath,
					timestamp: cache.Timestamp,
					size:      fileSize,
				})
			}

			// 如果总大小超过容量限制，按时间戳排序并删除最旧的文件
			if totalSize > int64(c.l3Capacity) {
				// 按时间戳排序（最旧的在前面）
				sort.Slice(cacheFiles, func(i, j int) bool {
					return cacheFiles[i].timestamp < cacheFiles[j].timestamp
				})

				// 删除最旧的文件，直到总大小低于容量限制
				for i := 0; i < len(cacheFiles) && totalSize > int64(c.l3Capacity); i++ {
					err := os.Remove(cacheFiles[i].path)
					if err != nil {
						return
					}
					totalSize -= cacheFiles[i].size
				}
			}
		}()
	}
}

// enforceL1CapacityLimit 强制执行L1缓存容量限制的内部方法
func (c *MultiLevelCache) enforceL1CapacityLimit() {
	for len(c.l1Cache) > c.l1Capacity {
		var oldestKey string
		var oldestTime int64 = math.MaxInt64
		for k, v := range c.l1Cache {
			if v.Timestamp < oldestTime {
				oldestTime = v.Timestamp
				oldestKey = k
			}
		}
		delete(c.l1Cache, oldestKey)
	}
}

// enforceL2CapacityLimit 强制执行L2缓存容量限制的内部方法
// 注意：此方法已被直接在各个操作中实现，保留此方法是为了兼容性
func (c *MultiLevelCache) enforceL2CapacityLimit() {
	// 使用共享内存时，容量限制在各个操作中直接处理
	if c.l2SharedMem != nil {
		// 检查共享内存中的条目数量是否超过容量限制
		if c.l2SharedMem.Count() > c.l2Capacity {
			// 获取所有键
			keys := c.l2SharedMem.GetAllKeys()

			// 如果键数量超过容量，删除一些旧的键
			if len(keys) > c.l2Capacity {
				// 删除超出部分的键
				for i := 0; i < len(keys)-c.l2Capacity; i++ {
					c.l2SharedMem.Delete(keys[i])
				}
			}
		}
	} else {
		// 如果共享内存不可用，使用普通内存作为fallback
		if c.l2FallbackCache == nil {
			c.l2FallbackCache = make(map[string]queryCache)
		}

		// 检查fallback缓存是否超过容量限制
		if len(c.l2FallbackCache) > c.l2Capacity {
			// 按时间戳排序，删除最旧的项
			type cacheItem struct {
				key       string
				timestamp int64
			}

			var items []cacheItem
			for k, v := range c.l2FallbackCache {
				items = append(items, cacheItem{key: k, timestamp: v.Timestamp})
			}

			// 按时间戳排序（最旧的在前面）
			sort.Slice(items, func(i, j int) bool {
				return items[i].timestamp < items[j].timestamp
			})

			// 删除超出容量的最旧项
			deleteCount := len(c.l2FallbackCache) - c.l2Capacity
			for i := 0; i < deleteCount; i++ {
				delete(c.l2FallbackCache, items[i].key)
			}
		}
	}
}

// CheckL1Cache 检查L1缓存
func (c *MultiLevelCache) CheckL1Cache(key string) (interface{}, bool) {
	c.l1Mu.RLock()
	defer c.l1Mu.RUnlock()

	if cache, found := c.l1Cache[key]; found && time.Now().Unix()-cache.Timestamp < 300 {
		c.statsMu.Lock()
		c.stats.L1Hits++
		c.statsMu.Unlock()
		return cache.Results, true
	}

	return nil, false
}

// CheckL2Cache 检查L2缓存
func (c *MultiLevelCache) CheckL2Cache(key string) (interface{}, bool) {
	c.l2Mu.RLock()
	defer c.l2Mu.RUnlock()

	if c.l2SharedMem != nil {
		// 使用共享内存
		results, timestamp, found := c.l2SharedMem.Get(key)
		if found && time.Now().Unix()-timestamp < 1800 {
			c.statsMu.Lock()
			c.stats.L2Hits++
			c.statsMu.Unlock()
			return results, true
		}
	} else if c.l2FallbackCache != nil {
		// 使用fallback缓存
		if cache, found := c.l2FallbackCache[key]; found && time.Now().Unix()-cache.Timestamp < 1800 {
			c.statsMu.Lock()
			c.stats.L2Hits++
			c.statsMu.Unlock()
			return cache.Results, true
		}
	}

	return nil, false
}

// CheckL3Cache 检查L3缓存
func (c *MultiLevelCache) CheckL3Cache(key string) (interface{}, bool) {
	c.l3Mu.RLock()
	defer c.l3Mu.RUnlock()

	cache, found := c.readFromL3Cache(key)
	if found {
		c.statsMu.Lock()
		c.stats.L3Hits++
		c.statsMu.Unlock()
		return cache.Results, true
	}

	return nil, false
}

// PromoteToL1Cache 将结果提升到L1缓存
func (c *MultiLevelCache) PromoteToL1Cache(key string, value interface{}) {
	c.l1Mu.Lock()
	defer c.l1Mu.Unlock()

	// 检测数据类型
	dataType := c.detectDataType(value)

	cache := queryCache{
		Results:   value,
		Timestamp: time.Now().Unix(),
		DataType:  dataType,
	}

	c.l1Cache[key] = cache

	// 如果L1缓存超出容量，移除最旧的项
	c.enforceL1CapacityLimit()
}

// PromoteToL2Cache 将结果提升到L2缓存
func (c *MultiLevelCache) PromoteToL2Cache(key string, value interface{}) {
	c.l2Mu.Lock()
	defer c.l2Mu.Unlock()

	// 检测数据类型
	dataType := c.detectDataType(value)
	timestamp := time.Now().Unix()

	if c.l2SharedMem != nil {
		// 根据数据类型处理L2缓存存储
		var err error
		switch dataType {
		case TypeStringArray:
			// 字符串数组类型，直接使用共享内存的Put方法
			if strArray, ok := value.([]string); ok {
				err = c.l2SharedMem.Put(key, strArray, timestamp)
			} else {
				err = fmt.Errorf("类型断言失败：预期[]string，实际%T", value)
			}
		default:
			// 其他类型需要先序列化为JSON字符串，然后存储
			data, jsonErr := json.Marshal(value)
			if jsonErr != nil {
				err = fmt.Errorf("序列化数据失败: %v", jsonErr)
			} else {
				// 将序列化后的数据作为字符串数组存储
				strData := []string{string(data), fmt.Sprintf("%d", dataType)}
				err = c.l2SharedMem.Put(key, strData, timestamp)
			}
		}

		if err != nil {
			// 记录错误，但不中断程序
			fmt.Printf("Error promoting to L2 shared memory: %v\n", err)
			return
		}

		// 检查共享内存中的条目数量是否超过容量限制
		if c.l2SharedMem.Count() > c.l2Capacity {
			// 获取所有键
			keys := c.l2SharedMem.GetAllKeys()

			// 如果键数量超过容量，删除一些旧的键
			if len(keys) > c.l2Capacity {
				// 删除超出部分的键
				for i := 0; i < len(keys)-c.l2Capacity; i++ {
					c.l2SharedMem.Delete(keys[i])
				}
			}
		}
	}
}

// Clear 清空缓存
func (c *MultiLevelCache) Clear() {
	// 清空L1缓存
	c.l1Mu.Lock()
	c.l1Cache = make(map[string]queryCache, c.l1Capacity)
	c.l1Mu.Unlock()

	// 清空L2缓存
	c.l2Mu.Lock()
	if c.l2SharedMem != nil {
		c.l2SharedMem.Clear()
	}
	if c.l2FallbackCache != nil {
		c.l2FallbackCache = make(map[string]queryCache)
	}
	c.l2Mu.Unlock()

	// 清空L3缓存
	if c.l3CachePath != "" {
		go func() {
			c.l3Mu.Lock()
			defer c.l3Mu.Unlock()

			// 遍历L3缓存目录
			files, err := fs.ReadDir(os.DirFS(c.l3CachePath), c.l3CachePath)
			if err != nil {
				fmt.Printf("Error reading L3 cache directory: %v\n", err)
				return
			}

			// 删除所有缓存文件
			for _, file := range files {
				if strings.HasSuffix(file.Name(), c.l3FileExt) {
					filePath := filepath.Join(c.l3CachePath, file.Name())
					err := os.Remove(filePath)
					if err != nil {
						return
					}
				}
			}
		}()
	}

	// 重置统计信息
	c.statsMu.Lock()
	c.stats = CacheStats{}
	c.statsMu.Unlock()
}

// PrewarmSystem 预热整个缓存系统
func (c *MultiLevelCache) PrewarmSystem() {
	// 这里实现预热缓存的逻辑
	// 例如，可以加载常用的查询结果到缓存中
	var err error
	// 预分配内存以提高性能
	c.l1Mu.Lock()
	c.l1Cache = make(map[string]queryCache, c.l1Capacity)
	c.l1Mu.Unlock()

	c.l2Mu.Lock()
	c.l2SharedMem, err = storage.NewSharedMemory()
	if err != nil {
		logger.Fatal("Error creating shared memory: %v\n", err)
	}
	c.l2Mu.Unlock()

	// 重置统计信息
	c.statsMu.Lock()
	c.stats = CacheStats{}
	c.statsMu.Unlock()

	// 从历史查询日志中加载热门查询
	if err := c.loadFromQueryLogs(); err != nil {
		logger.Error("从查询日志加载热门查询失败: %v", err)
	}

	// 从预定义的热门查询列表中加载数据
	if err := c.loadFromPredefinedQueries(); err != nil {
		logger.Error("从预定义查询列表加载数据失败: %v", err)
	}

	logger.Info("缓存预热完成，已加载热门查询到缓存")
}

// Prewarm 预热指定键值对的缓存
func (c *MultiLevelCache) Prewarm(key string, value interface{}) {
	// 检测数据类型
	//dataType := c.detectDataType(value)

	// 将数据提升到 L1 缓存
	c.PromoteToL1Cache(key, value)

	// 将数据提升到 L2 缓存
	c.PromoteToL2Cache(key, value)
}

// loadFromQueryLogs 从历史查询日志中加载热门查询
func (c *MultiLevelCache) loadFromQueryLogs() error {
	// 查询日志文件路径
	queryLogPath := filepath.Join(c.l3CachePath, "../logs/query_logs.json")

	// 检查文件是否存在
	if _, err := os.Stat(queryLogPath); os.IsNotExist(err) {
		logger.Info("查询日志文件不存在: %s", queryLogPath)
		return nil // 文件不存在不视为错误
	}

	// 读取查询日志文件
	data, err := fs.ReadFile(os.DirFS(queryLogPath), queryLogPath)
	if err != nil {
		return fmt.Errorf("读取查询日志文件失败: %v", err)
	}

	// 解析查询日志
	type QueryLogEntry struct {
		Query     string   `json:"query"`
		Results   []string `json:"results"`
		Timestamp int64    `json:"timestamp"`
		HitCount  int      `json:"hit_count"`
	}

	var queryLogs []QueryLogEntry
	if err := json.Unmarshal(data, &queryLogs); err != nil {
		return fmt.Errorf("解析查询日志失败: %v", err)
	}

	// 按照访问次数排序
	sort.Slice(queryLogs, func(i, j int) bool {
		return queryLogs[i].HitCount > queryLogs[j].HitCount
	})

	// 加载前N个热门查询到缓存
	maxQueries := 100 // 最多加载100个热门查询
	if len(queryLogs) < maxQueries {
		maxQueries = len(queryLogs)
	}

	loadedCount := 0
	for i := 0; i < maxQueries; i++ {
		entry := queryLogs[i]
		// 只加载有结果且在一周内的查询
		if len(entry.Results) > 0 && time.Now().Unix()-entry.Timestamp < 7*24*60*60 {
			// 检测数据类型
			dataType := c.detectDataType(entry.Results)

			// 构造缓存对象
			cache := queryCache{
				Results:   entry.Results,
				Timestamp: entry.Timestamp,
				DataType:  dataType,
			}

			// 存入L1缓存
			c.l1Mu.Lock()
			c.l1Cache[entry.Query] = cache
			c.l1Mu.Unlock()

			// 存入L2缓存
			c.l2Mu.Lock()
			if c.l2SharedMem != nil {
				// 根据数据类型处理L2缓存存储
				var err error
				switch dataType {
				case TypeStringArray:
					// 字符串数组类型，直接使用共享内存的Put方法
					err = c.l2SharedMem.Put(entry.Query, entry.Results, entry.Timestamp)
				default:
					// 其他类型需要先序列化为JSON字符串，然后存储
					data, jsonErr := json.Marshal(entry.Results)
					if jsonErr != nil {
						err = fmt.Errorf("序列化数据失败: %v", jsonErr)
					} else {
						// 将序列化后的数据作为字符串数组存储
						strData := []string{string(data), fmt.Sprintf("%d", dataType)}
						err = c.l2SharedMem.Put(entry.Query, strData, entry.Timestamp)
					}
				}

				if err != nil {
					// 记录错误，但不中断程序
					fmt.Printf("Error writing to L2 shared memory: %v\n", err)
				}
			}
			c.l2Mu.Unlock()

			loadedCount++
		}
	}

	logger.Info("从查询日志加载了 %d 个热门查询到缓存", loadedCount)
	return nil
}

// loadFromPredefinedQueries 从预定义的热门查询列表中加载数据
func (c *MultiLevelCache) loadFromPredefinedQueries() error {
	// 预定义的热门查询列表文件路径
	predefinedQueriesPath := filepath.Join(c.l3CachePath, "../config/popular_queries.json")

	// 检查文件是否存在
	if _, err := os.Stat(predefinedQueriesPath); os.IsNotExist(err) {
		logger.Info("预定义热门查询文件不存在: %s", predefinedQueriesPath)
		return nil // 文件不存在不视为错误
	}

	// 读取预定义热门查询文件
	data, err := fs.ReadFile(os.DirFS(predefinedQueriesPath), predefinedQueriesPath)
	if err != nil {
		return fmt.Errorf("读取预定义热门查询文件失败: %v", err)
	}

	// 解析预定义热门查询
	type PredefinedQuery struct {
		Query   string   `json:"query"`
		Results []string `json:"results"`
	}

	var predefinedQueries []PredefinedQuery
	if err := json.Unmarshal(data, &predefinedQueries); err != nil {
		return fmt.Errorf("解析预定义热门查询失败: %v", err)
	}

	// 加载预定义热门查询到缓存
	loadedCount := 0
	timestamp := time.Now().Unix()

	for _, query := range predefinedQueries {
		if len(query.Results) > 0 {
			// 检测数据类型
			dataType := c.detectDataType(query.Results)

			// 构造缓存对象
			cache := queryCache{
				Results:   query.Results,
				Timestamp: timestamp,
				DataType:  dataType,
			}

			// 存入L1缓存
			c.l1Mu.Lock()
			c.l1Cache[query.Query] = cache
			c.l1Mu.Unlock()

			// 存入L2缓存
			c.l2Mu.Lock()
			if c.l2SharedMem != nil {
				// 根据数据类型处理L2缓存存储
				var err error
				switch dataType {
				case TypeStringArray:
					// 字符串数组类型，直接使用共享内存的Put方法
					err = c.l2SharedMem.Put(query.Query, query.Results, timestamp)
				default:
					// 其他类型需要先序列化为JSON字符串，然后存储
					data, jsonErr := json.Marshal(query.Results)
					if jsonErr != nil {
						err = fmt.Errorf("序列化数据失败: %v", jsonErr)
					} else {
						// 将序列化后的数据作为字符串数组存储
						strData := []string{string(data), fmt.Sprintf("%d", dataType)}
						err = c.l2SharedMem.Put(query.Query, strData, timestamp)
					}
				}

				if err != nil {
					// 记录错误，但不中断程序
					fmt.Printf("Error writing to L2 shared memory: %v\n", err)
				}
			}
			c.l2Mu.Unlock()

			loadedCount++
		}
	}

	logger.Info("从预定义列表加载了 %d 个热门查询到缓存", loadedCount)
	return nil
}

// 生成L3缓存文件路径
func (c *MultiLevelCache) getL3CacheFilePath(key string) string {
	// 处理空键的特殊情况
	if key == "" {
		key = "__empty_key__"
	}

	// 使用MD5哈希作为文件名，避免特殊字符和路径过长问题
	hashKey := fmt.Sprintf("%x", md5.Sum([]byte(key)))
	return filepath.Join(c.l3CachePath, hashKey+c.l3FileExt)
}

// 将缓存数据序列化为JSON
func (c *MultiLevelCache) serializeCache(cache queryCache) ([]byte, error) {
	// 确保DataType字段已设置
	if cache.DataType == TypeUnknown {
		cache.DataType = c.detectDataType(cache.Results)
	}
	return json.Marshal(cache)
}

// 从JSON反序列化缓存数据
func (c *MultiLevelCache) deserializeCache(data []byte) (queryCache, error) {
	var cache queryCache
	err := json.Unmarshal(data, &cache)

	// 处理旧版本缓存数据兼容性
	if cache.DataType == TypeUnknown && cache.Results != nil {
		cache.DataType = c.detectDataType(cache.Results)
	}

	return cache, err
}

// 检测数据类型
func (c *MultiLevelCache) detectDataType(data interface{}) CacheDataType {
	if data == nil {
		return TypeUnknown
	}

	switch data.(type) {
	case []string:
		return TypeStringArray
	case []float64:
		return TypeFloat64Array
	case []int:
		return TypeIntArray
	case map[string]interface{}:
		return TypeMap
	case []byte:
		return TypeBytes
	default:
		// 使用反射检查类型
		rt := reflect.TypeOf(data)
		if rt == nil {
			return TypeUnknown
		}

		// 检查是否为结构体
		if rt.Kind() == reflect.Struct {
			return TypeStruct
		}

		// 检查是否为指针类型，如果是，获取其指向的类型
		if rt.Kind() == reflect.Ptr {
			elem := rt.Elem()
			if elem.Kind() == reflect.Struct {
				return TypeStruct
			}
		}

		// 检查是否为切片类型
		if rt.Kind() == reflect.Slice {
			elem := rt.Elem()
			switch elem.Kind() {
			case reflect.String:
				return TypeStringArray
			case reflect.Float64:
				return TypeFloat64Array
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				return TypeIntArray
			case reflect.Uint8: // []byte
				return TypeBytes
			default:
				// 未知的切片元素类型，返回TypeUnknown
				return TypeUnknown
			}
		}

		// 检查是否为映射类型
		if rt.Kind() == reflect.Map {
			if rt.Key().Kind() == reflect.String {
				return TypeMap
			}
		}

		return TypeUnknown
	}
}

// deleteFromL3Cache 从L3磁盘缓存删除指定键的缓存文件
func (c *MultiLevelCache) deleteFromL3Cache(key string) {
	if c.l3CachePath == "" {
		return
	}

	filePath := c.getL3CacheFilePath(key)

	// 检查文件是否存在
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return
	}

	// 删除文件
	err := os.Remove(filePath)
	if err != nil {
		fmt.Printf("Error deleting L3 cache file %s: %v\n", filePath, err)
	}
}

// 读取数据从L3磁盘缓存
func (c *MultiLevelCache) readFromL3Cache(key string) (queryCache, bool) {
	if c.l3CachePath == "" {
		return queryCache{}, false
	}

	filePath := c.getL3CacheFilePath(key)

	// 检查文件是否存在
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return queryCache{}, false
	}

	// 读取文件内容
	data, err := os.ReadFile(filePath)
	if err != nil {
		return queryCache{}, false
	}

	// 反序列化
	cache, err := c.deserializeCache(data)
	if err != nil {
		return queryCache{}, false
	}

	// 检查是否过期
	if time.Now().Unix()-cache.Timestamp > int64(c.l3TTL.Seconds()) {
		// 异步删除过期文件
		go func() {
			err := os.Remove(filePath)
			if err != nil {
				logger.Error("remove %s file error: %v", filePath, err)
			}
		}()
		return queryCache{}, false
	}

	return cache, true
}

// 写入数据到L3磁盘缓存
func (c *MultiLevelCache) writeToL3Cache(key string, cache queryCache) error {
	if c.l3CachePath == "" {
		return nil
	}

	// 确保L3缓存目录存在
	if err := os.MkdirAll(c.l3CachePath, 0755); err != nil {
		// 如果目录创建失败，禁用L3缓存并返回nil而不是错误
		c.l3CachePath = ""
		return nil
	}

	// 确保设置了数据类型
	if cache.DataType == TypeUnknown {
		cache.DataType = c.detectDataType(cache.Results)
	}

	// 序列化缓存数据
	data, err := c.serializeCache(cache)
	if err != nil {
		return err
	}

	// 写入文件
	filePath := c.getL3CacheFilePath(key)
	return os.WriteFile(filePath, data, 0644)
}

// getAvailableDiskSpace 获取指定路径的可用磁盘空间
// 这个函数会根据操作系统自动选择合适的实现方法
func (c *MultiLevelCache) getAvailableDiskSpace(path string) (int64, error) {
	// 获取文件系统统计信息
	var stat os.FileInfo
	var err error
	
	// 确保路径存在
	if stat, err = os.Stat(path); err != nil {
		return 0, fmt.Errorf("无法访问路径 %s: %v", path, err)
	}
	
	// 如果不是目录，获取父目录
	if !stat.IsDir() {
		path = filepath.Dir(path)
	}
	
	// 根据操作系统使用不同的API获取磁盘空间
	// 平台特定的实现在对应的构建标签文件中定义
	switch runtime.GOOS {
	case "windows":
		// Windows系统使用Windows API (GetDiskFreeSpaceEx)
		return c.getAvailableDiskSpaceWindows(path)
	case "linux", "darwin", "freebsd", "openbsd", "netbsd":
		// Unix系统使用syscall.Statfs
		return c.getAvailableDiskSpaceUnix(path)
	default:
		// 对于不支持的操作系统，使用fallback方法
		fmt.Printf("Warning: 不支持的操作系统 %s，使用fallback方法\n", runtime.GOOS)
		return c.getAvailableDiskSpaceFallback(path)
	}
}

// getAvailableDiskSpaceWindows 在Windows系统上获取可用磁盘空间
func (c *MultiLevelCache) getAvailableDiskSpaceWindows(path string) (int64, error) {
	// 在Windows系统上，调用平台特定的实现
	if runtime.GOOS == "windows" {
		// 直接调用Windows平台特定的实现
		// 注意：getAvailableDiskSpaceWindowsNative方法在disk_space_windows.go中定义
		// 只有在Windows环境下编译时才会包含该文件
		return c.getAvailableDiskSpaceWindowsNative(path)
	}
	
	// 如果不是Windows系统（这种情况不应该发生，因为调用前已经检查了系统类型）
	// 但为了代码健壮性，我们提供一个fallback
	fmt.Printf("Warning: 在非Windows系统上调用了Windows磁盘空间检测函数\n")
	return c.getAvailableDiskSpaceFallback(path)
}

// getAvailableDiskSpaceUnix 在Unix系统（Linux、macOS等）上获取可用磁盘空间
// 注意：这个函数在Unix系统编译时会被disk_space_unix.go中的同名函数覆盖
func (c *MultiLevelCache) getAvailableDiskSpaceUnix(path string) (int64, error) {
	// 在Windows环境下编译时，这个函数会被调用
	// 但实际的Unix实现在disk_space_unix.go中（仅在Unix系统编译时包含）
	if runtime.GOOS == "windows" {
		fmt.Printf("Warning: 在Windows系统上调用了Unix磁盘空间检测函数\n")
		return c.getAvailableDiskSpaceFallback(path)
	}
	
	// 这个分支在Windows编译时不会被执行，但为了代码完整性保留
	// 在真正的Unix环境下编译时，disk_space_unix.go中的实现会覆盖这个函数
	fmt.Printf("Info: 使用主文件中的Unix磁盘空间检测fallback实现\n")
	return c.getAvailableDiskSpaceFallback(path)
}

// getAvailableDiskSpaceFallback 备用方法，用于不支持的操作系统或系统调用失败时
// 
// 使用第三方库的示例（可选）：
// 如果需要更准确的磁盘空间信息，可以使用github.com/shirou/gopsutil/v3库：
// 
// import "github.com/shirou/gopsutil/v3/disk"
// 
// func (c *MultiLevelCache) getAvailableDiskSpaceWithGopsutil(path string) (int64, error) {
//     usage, err := disk.Usage(path)
//     if err != nil {
//         return 0, err
//     }
//     return int64(usage.Free), nil
// }
//
func (c *MultiLevelCache) getAvailableDiskSpaceFallback(path string) (int64, error) {
	// 注意：这是一个基础的fallback实现
	// 生产环境中建议使用github.com/shirou/gopsutil/v3库获取更准确的磁盘空间信息
	
	// 尝试创建一个临时文件来测试写入权限和基本可用性
	tempFile := filepath.Join(path, ".temp_space_check")
	
	// 尝试创建一个小的测试文件
	file, err := os.Create(tempFile)
	if err != nil {
		return 0, fmt.Errorf("无法在路径 %s 创建测试文件: %v", path, err)
	}
	file.Close()
	
	// 立即删除测试文件
	os.Remove(tempFile)
	
	// 尝试通过文件系统的一些启发式方法来估算可用空间
	// 这是一个非常基础的估算方法
	availableSpace := c.estimateDiskSpaceHeuristic(path)
	
	fmt.Printf("Warning: 使用fallback方法获取磁盘空间，操作系统: %s，估算可用空间: %.2f GB\n", 
		runtime.GOOS, float64(availableSpace)/(1024*1024*1024))
	return availableSpace, nil
}

// estimateDiskSpaceHeuristic 使用启发式方法估算磁盘可用空间
func (c *MultiLevelCache) estimateDiskSpaceHeuristic(path string) int64 {
	// 获取路径信息
	stat, err := os.Stat(path)
	if err != nil {
		// 如果无法获取路径信息，返回保守估计
		return 512 * 1024 * 1024 // 512MB
	}
	
	// 如果是文件，获取其父目录
	if !stat.IsDir() {
		path = filepath.Dir(path)
	}
	
	// 尝试创建一系列测试文件来估算可用空间
	// 这是一个粗略的方法，不推荐在生产环境中使用
	testSizes := []int64{
		1024 * 1024,      // 1MB
		10 * 1024 * 1024, // 10MB
		100 * 1024 * 1024, // 100MB
	}
	
	var maxWritableSize int64 = 0
	
	for _, size := range testSizes {
		testFile := filepath.Join(path, fmt.Sprintf(".temp_size_test_%d", size))
		
		// 尝试创建指定大小的文件
		file, err := os.Create(testFile)
		if err != nil {
			break
		}
		
		// 尝试写入数据
		data := make([]byte, 1024) // 1KB块
		var written int64 = 0
		
		for written < size {
			n, err := file.Write(data)
			if err != nil {
				break
			}
			written += int64(n)
			if written >= size {
				maxWritableSize = size
				break
			}
		}
		
		file.Close()
		os.Remove(testFile) // 立即删除测试文件
		
		if written < size {
			break
		}
	}
	
	// 基于测试结果估算可用空间
	if maxWritableSize > 0 {
		// 假设实际可用空间是测试成功的最大文件大小的10倍
		return maxWritableSize * 10
	}
	
	// 如果所有测试都失败，返回最小估计值
	return 256 * 1024 * 1024 // 256MB
}

// updateL3CapacityConfig 更新L3缓存容量配置记录
func (c *MultiLevelCache) updateL3CapacityConfig() {
	if c.l3CachePath == "" {
		return
	}
	
	// 创建配置记录文件
	configFile := filepath.Join(c.l3CachePath, ".cache_config")
	
	// 准备配置信息
	configData := map[string]interface{}{
		"capacity":    c.l3Capacity,
		"updated_at":  time.Now().Unix(),
		"cache_path": c.l3CachePath,
		"file_ext":   c.l3FileExt,
		"ttl_hours":  int(c.l3TTL.Hours()),
	}
	
	// 序列化配置数据
	data, err := json.MarshalIndent(configData, "", "  ")
	if err != nil {
		fmt.Printf("Warning: Failed to serialize L3 cache config: %v\n", err)
		return
	}
	
	// 写入配置文件
	err = os.WriteFile(configFile, data, 0644)
	if err != nil {
		fmt.Printf("Warning: Failed to write L3 cache config file: %v\n", err)
	} else {
		fmt.Printf("L3 cache configuration updated in %s\n", configFile)
	}
}
