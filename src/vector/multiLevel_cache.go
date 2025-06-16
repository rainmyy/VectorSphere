package vector

import (
	"VectorSphere/src/library/logger"
	"VectorSphere/src/library/storage"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
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
	l2Capacity  int
	l2Mu        sync.RWMutex
	l2SharedMem *storage.SharedMemory // 共享内存管理器

	// L3: 磁盘缓存 - 较慢，容量大
	l3CachePath string
	l3Capacity  int
	l3FileExt   string
	l3TTL       time.Duration
	l3Mu        sync.RWMutex

	// 缓存统计
	stats   CacheStats
	statsMu sync.RWMutex
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
	// 确保L3缓存目录存在
	if l3Path != "" {
		os.MkdirAll(l3Path, 0755)
	}

	cache := &MultiLevelCache{
		l1Cache:     make(map[string]queryCache, l1Capacity),
		l1Capacity:  l1Capacity,
		l2Capacity:  l2Capacity,
		l3CachePath: l3Path,
		l3Capacity:  l3Capacity,
		l3FileExt:   ".cache",
		l3TTL:       24 * time.Hour, // 默认L3缓存过期时间为24小时
		stats:       CacheStats{},
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
		return cache.Results, true
	}

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
		if v.Timestamp < timestamp {
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
			if found && entryTimestamp < timestamp {
				c.l2SharedMem.Delete(key)
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
			files, err := ioutil.ReadDir(c.l3CachePath)
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
				data, err := ioutil.ReadFile(filePath)
				if err != nil {
					continue
				}

				// 反序列化
				cache, err := c.deserializeCache(data)
				if err != nil {
					// 无法解析的文件直接删除
					os.Remove(filePath)
					continue
				}

				// 检查是否过期
				if cache.Timestamp < timestamp {
					os.Remove(filePath)
				}
			}
		}()
	}
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
			files, err := ioutil.ReadDir(c.l3CachePath)
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
			cacheFiles := []cacheFileInfo{}
			totalSize := int64(0)

			for _, file := range files {
				if !strings.HasSuffix(file.Name(), c.l3FileExt) {
					continue
				}

				filePath := filepath.Join(c.l3CachePath, file.Name())
				fileSize := file.Size()
				totalSize += fileSize

				// 读取文件内容获取时间戳
				data, err := ioutil.ReadFile(filePath)
				if err != nil {
					continue
				}

				// 反序列化
				cache, err := c.deserializeCache(data)
				if err != nil {
					// 无法解析的文件直接删除
					os.Remove(filePath)
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
					os.Remove(cacheFiles[i].path)
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
	c.l2Mu.Unlock()

	// 清空L3缓存
	if c.l3CachePath != "" {
		go func() {
			c.l3Mu.Lock()
			defer c.l3Mu.Unlock()

			// 遍历L3缓存目录
			files, err := ioutil.ReadDir(c.l3CachePath)
			if err != nil {
				fmt.Printf("Error reading L3 cache directory: %v\n", err)
				return
			}

			// 删除所有缓存文件
			for _, file := range files {
				if strings.HasSuffix(file.Name(), c.l3FileExt) {
					filePath := filepath.Join(c.l3CachePath, file.Name())
					os.Remove(filePath)
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
	data, err := ioutil.ReadFile(queryLogPath)
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
	data, err := ioutil.ReadFile(predefinedQueriesPath)
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

// 从L3磁盘缓存读取数据
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
	data, err := ioutil.ReadFile(filePath)
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
		go os.Remove(filePath)
		return queryCache{}, false
	}

	return cache, true
}

// 写入数据到L3磁盘缓存
func (c *MultiLevelCache) writeToL3Cache(key string, cache queryCache) error {
	if c.l3CachePath == "" {
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
	return ioutil.WriteFile(filePath, data, 0644)
}
